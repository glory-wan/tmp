from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def patch_huggingface_hub_for_diffusers_024() -> None:
    """Keep diffusers==0.24 usable with newer huggingface-hub without changing SD code."""
    try:
        import huggingface_hub.constants as constants
        if not hasattr(constants, "hf_cache_home"):
            constants.hf_cache_home = constants.HF_HOME
    except Exception:
        pass


def setup_logger(output_dir: str | Path) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("detection_depth_adapt")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(Path(output_dir) / "pipeline.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def run_logged_subprocess(command: list[str], cwd: str | Path, logger: logging.Logger,
                          log_file: str | Path, prefix: str = "subprocess") -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("command: %s", " ".join(command))
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=Path(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            handle.write(line + "\n")
            handle.flush()
            logger.info("[%s] %s", prefix, line)
        code = process.wait()
        if code:
            logger.error("%s failed with exit code %d", prefix, code)
            raise subprocess.CalledProcessError(code, command)


def parse_class_subset(enabled: bool = False, count: int | None = None,
                       class_ids: str | list[int] | None = None) -> set[int] | None:
    if not enabled:
        return None
    if class_ids:
        if isinstance(class_ids, str):
            return {int(x) for x in class_ids.split(",") if x.strip()}
        return {int(x) for x in class_ids}
    if count is None:
        count = 10
    return set(range(int(count)))


def subset_label(class_subset: set[int] | None) -> str:
    if class_subset is None:
        return "full"
    return ",".join(str(x) for x in sorted(class_subset))


class CocoMini:
    def __init__(self, annotation_file: str | Path, images_dir: str | Path, class_subset: set[int] | None = None):
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.class_subset = set(class_subset) if class_subset is not None else None
        data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.images = {int(x["id"]): x for x in data["images"]}
        self.categories = {int(x["id"]): x for x in data["categories"]}
        ordered = sorted(self.categories)
        self.category_to_class = {cat_id: idx for idx, cat_id in enumerate(ordered)}
        self.class_to_category = {v: k for k, v in self.category_to_class.items()}
        self.class_names = [self.categories[x]["name"] for x in ordered]
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) or ann.get("area", 0) <= 0:
                continue
            item = dict(ann)
            item["class_id"] = self.category_to_class[int(ann["category_id"])]
            if self.class_subset is not None and item["class_id"] not in self.class_subset:
                continue
            self.annotations[int(ann["image_id"])].append(item)
        if self.class_subset is not None:
            valid_ids = {image_id for image_id, anns in self.annotations.items() if anns}
            self.images = {image_id: info for image_id, info in self.images.items() if image_id in valid_ids}

    def image_path(self, image_id: int) -> Path:
        file_name = Path(self.images[int(image_id)]["file_name"])
        return file_name if file_name.is_absolute() else self.images_dir / file_name

    def iter_images(self, limit: int | None = None):
        ids = sorted(self.images)
        if limit is not None:
            ids = ids[:limit]
        for image_id in ids:
            path = self.image_path(image_id)
            if path.exists():
                yield image_id, path, self.annotations.get(image_id, [])

    def export_yolo(self, output: str | Path, image_ids: list[int]) -> Path:
        output = Path(output)
        images_out, labels_out = output / "images", output / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        kept = []
        for image_id in image_ids:
            info = self.images[int(image_id)]
            src = self.image_path(image_id)
            if not src.exists():
                continue
            dst = images_out / info["file_name"]
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            lines = []
            width, height = float(info["width"]), float(info["height"])
            for ann in self.annotations.get(int(image_id), []):
                x, y, w, h = map(float, ann["bbox"])
                lines.append(f"{ann['class_id']} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}")
            (labels_out / (Path(info["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            kept.append(str(dst.absolute()))
        list_path = output / "images.txt"
        list_path.write_text("\n".join(kept) + "\n")
        return list_path

    def export_coco_annotations(self, output: str | Path, image_ids: list[int] | None = None) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image_set = set(self.images) if image_ids is None else {int(x) for x in image_ids}
        images = [self.images[x] for x in sorted(image_set) if x in self.images]
        annotations = []
        for image_id in sorted(image_set):
            for ann in self.annotations.get(image_id, []):
                item = dict(ann)
                item.pop("class_id", None)
                annotations.append(item)
        categories = [self.categories[x] for x in sorted(self.categories)]
        output.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, indent=2), encoding="utf-8")
        return output


@dataclass
class FailureSample:
    image_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    failure_type: str
    confidence: float
    top2_gap: float = 1.0
    forgotten_count: int = 0
    source: str = "real"


def save_failures(path: str | Path, failures: list[FailureSample], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "failures": [asdict(x) for x in failures]}, indent=2), encoding="utf-8")


def load_failures(path: str | Path, class_subset: set[int] | None = None) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    failures = data["failures"]
    if any(x.get("source") != "real" for x in failures):
        raise ValueError("Failure pool must contain real-image instances only.")
    if class_subset is not None:
        failures = [x for x in failures if int(x["class_id"]) in class_subset]
    return failures


def prompt_group_key(class_id: int, failure_type: str, scope: str) -> str:
    return f"class_{class_id}" if scope == "class" else f"class_{class_id}_{failure_type}"


def placeholder_tokens(group: str, n: int) -> list[str]:
    return [f"<det_{group}_{i}>" for i in range(n)]


def bbox_mask(size: tuple[int, int], bbox, padding: float = 0.0) -> Image.Image:
    x, y, w, h = bbox
    px, py = w * padding, h * padding
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((max(0, x - px), max(0, y - py), min(size[0], x + w + px), min(size[1], y + h + py)), fill=255)
    return mask


def load_depth_style_components(model_name_or_path: str, dtype=torch.float16, local_files_only: bool = True,
                                text_encoder_dtype=None):
    """Load SD inpainting components explicitly, following the depth scripts' component style."""
    patch_huggingface_hub_for_diffusers_024()
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path, subfolder="tokenizer", local_files_only=local_files_only)
    text_encoder_kwargs = {"local_files_only": local_files_only}
    if text_encoder_dtype is not None:
        text_encoder_kwargs["torch_dtype"] = text_encoder_dtype
    text_encoder = CLIPTextModel.from_pretrained(model_name_or_path, subfolder="text_encoder", **text_encoder_kwargs)
    vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only)
    unet = UNet2DConditionModel.from_pretrained(model_name_or_path, subfolder="unet", torch_dtype=dtype, local_files_only=local_files_only)
    scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler", local_files_only=local_files_only)
    return tokenizer, text_encoder, vae, unet, scheduler


def add_placeholder_tokens(tokenizer, text_encoder, groups: dict[str, list[str]], init_mode: str, initializer_token: str | None = None) -> dict[str, list[int]]:
    token_list = [token for tokens in groups.values() for token in tokens]
    original_vocab = len(tokenizer)
    added = tokenizer.add_tokens(token_list)
    if added != len(token_list):
        raise ValueError("Placeholder token collision; use a fresh tokenizer/model instance.")
    token_ids = {group: tokenizer.convert_tokens_to_ids(tokens) for group, tokens in groups.items()}
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        if init_mode == "mean_emb_cov_emb":
            base = token_embeds[:original_vocab].detach().float().cpu().numpy()
            weights = np.random.multivariate_normal(mean=base.mean(axis=0), cov=np.cov(base, rowvar=0), size=len(token_list))
            for token, weight in zip(token_list, weights):
                token_embeds[tokenizer.convert_tokens_to_ids(token)] = torch.tensor(weight, dtype=token_embeds.dtype, device=token_embeds.device)
        else:
            if initializer_token is None:
                initializer_token = "object"
            init_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
            if len(init_ids) != 1:
                raise ValueError("initializer_token must map to exactly one CLIP token.")
            for token in token_list:
                token_embeds[tokenizer.convert_tokens_to_ids(token)] = token_embeds[init_ids[0]]
    return token_ids


def tensor_from_image(path: Path, resolution: int, device, dtype=torch.float32):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image.resize((resolution, resolution))).copy()
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype) / 127.5 - 1.0
    return image, tensor


def mask_tensor_from_bbox(image_size: tuple[int, int], bbox, resolution: int, device, dtype=torch.float32):
    mask = bbox_mask(image_size, bbox).resize((resolution, resolution))
    arr = np.asarray(mask).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

# 多尺度 ROI cosine similarity loss + 像素 L1约束
def roi_semantic_loss(generated, original, mask):
    losses = []
    for scale in (1, 4, 16):
        gen = F.avg_pool2d(generated * mask, scale, scale).flatten(1)
        ref = F.avg_pool2d(original * mask, scale, scale).flatten(1)
        losses.append(1 - F.cosine_similarity(gen, ref).mean()) 
        #把目标区域分别：原尺寸/下采样4倍/下采样16倍；然后比较方向相似度。
        #L1约束-限制生成区域不要完全变化
    return sum(losses) / len(losses) + 0.1 * ((generated - original).abs() * mask).sum() / mask.sum().clamp_min(1)


def write_yolo_yaml(path: str | Path, train_list: Path, val_list: Path, names: list[str]) -> None:
    import yaml
    payload = {"train": str(train_list), "val": str(val_list), "nc": len(names), "names": names}
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
