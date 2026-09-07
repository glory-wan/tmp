from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| (DEBUG|INFO|WARNING|ERROR|CRITICAL) \| (.*)$")


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
    logger = logging.getLogger("detection_gligen_sdedit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(Path(output_dir) / "pipeline.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    if os.environ.get("DETECTION_GLIGEN_SDEDIT_FILE_LOG_ONLY") != "1":
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def normalize_subprocess_log_line(line: str) -> tuple[int, str]:
    match = LOG_LINE_RE.match(line)
    if not match:
        return logging.INFO, line
    level_name, message = match.groups()
    return logging.getLevelName(level_name), message


def yolov7_device_arg(device: str | int | None) -> str:
    """Return a YOLOv7 --device value without breaking CUDA_VISIBLE_DEVICES remapping.

    Official YOLOv7 calls ``os.environ["CUDA_VISIBLE_DEVICES"] = device`` when
    a non-empty device is passed. If the parent process already set
    CUDA_VISIBLE_DEVICES=7, passing --device 0 would silently remap the child
    back to physical GPU 0. Returning an empty string lets YOLOv7 preserve the
    parent's visible-device mapping and still use cuda:0 inside that view.
    """
    value = "" if device is None else str(device)
    if value.lower() == "cpu":
        return value
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1" and value in {"", "0", "cuda:0"}:
        return ""
    return value


def append_yolov7_device_arg(command: list[str], device: str | int | None) -> None:
    value = yolov7_device_arg(device)
    if value:
        command.extend(["--device", value])


def run_logged_subprocess(command: list[str], cwd: str | Path, logger: logging.Logger,
                          log_file: str | Path, prefix: str = "subprocess",
                          env: dict[str, str] | None = None) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("command: %s", " ".join(command))
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=Path(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            handle.write(line + "\n")
            handle.flush()
            if not line:
                continue
            level, message = normalize_subprocess_log_line(line)
            logger.log(level, "[%s] %s", prefix, message)
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


def normalize_class_name(value: str) -> str:
    """Normalize a category name for YAML-to-COCO matching."""

    return " ".join(str(value).strip().casefold().split())


def load_dataset_yaml_names(dataset_yaml: str | Path) -> list[str]:
    """Load an Ultralytics dataset YAML and return names in model-id order."""

    import yaml

    path = Path(dataset_yaml).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        try:
            indexed = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset YAML names keys must be integer class ids: {path}") from exc
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError(
                f"Dataset YAML names ids must be contiguous from 0; got {sorted(indexed)} in {path}"
            )
        ordered_names = [indexed[index] for index in expected]
    elif isinstance(names, (list, tuple)):
        ordered_names = [str(value) for value in names]
    else:
        raise ValueError(f"Dataset YAML must define names as a list or id-to-name mapping: {path}")
    if not ordered_names or any(not name.strip() for name in ordered_names):
        raise ValueError(f"Dataset YAML names must contain at least one non-empty class name: {path}")
    normalized = [normalize_class_name(name) for name in ordered_names]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Dataset YAML contains duplicate class names after normalization: {duplicates}")
    return ordered_names


class CocoMini:
    def __init__(
        self,
        annotation_file: str | Path,
        images_dir: str | Path,
        class_subset: set[int] | None = None,
        dataset_yaml: str | Path | None = None,
    ):
        self.annotation_file = Path(annotation_file)
        self.images_dir = Path(images_dir)
        self.class_subset = set(class_subset) if class_subset is not None else None
        self.dataset_yaml = Path(dataset_yaml).expanduser().resolve() if dataset_yaml is not None else None
        data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.images = {int(x["id"]): x for x in data["images"]}
        source_categories = {int(x["id"]): x for x in data["categories"]}
        if self.dataset_yaml is not None:
            model_names = load_dataset_yaml_names(self.dataset_yaml)
            categories_by_name: dict[str, int] = {}
            for category_id, category in source_categories.items():
                normalized = normalize_class_name(category["name"])
                if normalized in categories_by_name:
                    other = categories_by_name[normalized]
                    raise ValueError(
                        f"COCO annotations contain duplicate normalized category name {normalized!r}: "
                        f"category_id={other} and category_id={category_id}"
                    )
                categories_by_name[normalized] = category_id
            missing = [
                name for name in model_names
                if normalize_class_name(name) not in categories_by_name
            ]
            if missing:
                raise ValueError(
                    f"Dataset YAML classes are missing from COCO categories: {missing}. "
                    f"dataset_yaml={self.dataset_yaml} annotations={self.annotation_file}"
                )
            ordered = [categories_by_name[normalize_class_name(name)] for name in model_names]
            self.class_names = model_names
        else:
            ordered = sorted(source_categories)
            self.class_names = [str(source_categories[category_id]["name"]) for category_id in ordered]
        self.categories = {category_id: source_categories[category_id] for category_id in ordered}
        self.category_to_class = {cat_id: idx for idx, cat_id in enumerate(ordered)}
        self.class_to_category = {v: k for k, v in self.category_to_class.items()}
        if self.class_subset is not None:
            invalid = sorted(self.class_subset - set(range(len(self.class_names))))
            if invalid:
                source = f"dataset YAML {self.dataset_yaml}" if self.dataset_yaml is not None else "COCO categories"
                raise ValueError(
                    f"class_subset contains ids outside the {len(self.class_names)}-class space from {source}: {invalid}"
                )
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) or ann.get("area", 0) <= 0:
                continue
            category_id = int(ann["category_id"])
            if category_id not in self.category_to_class:
                continue
            item = dict(ann)
            item["class_id"] = self.category_to_class[category_id]
            if self.class_subset is not None and item["class_id"] not in self.class_subset:
                continue
            self.annotations[int(ann["image_id"])].append(item)
        if self.class_subset is not None or self.dataset_yaml is not None:
            valid_ids = {image_id for image_id, anns in self.annotations.items() if anns}
            self.images = {image_id: info for image_id, info in self.images.items() if image_id in valid_ids}

    @property
    def class_mapping(self) -> list[dict]:
        return [
            {
                "model_class_id": class_id,
                "class_name": self.class_names[class_id],
                "coco_category_id": self.class_to_category[class_id],
            }
            for class_id in range(len(self.class_names))
        ]

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
        categories = [self.categories[self.class_to_category[class_id]] for class_id in sorted(self.class_to_category)]
        output.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, indent=2), encoding="utf-8")
        return output


@dataclass
class FailureSample:
    image_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    failure_type: str
    confidence: float
    top2_gap: float | None = None
    forgotten_count: int = 0
    annotation_id: int | None = None
    source: str = "real"


def save_failures(path: str | Path, failures: list[FailureSample], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "failures": [asdict(x) for x in failures]}, indent=2), encoding="utf-8")


def load_failures(
    path: str | Path,
    class_subset: set[int] | None = None,
    expected_class_names: list[str] | None = None,
) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    saved_names = data.get("meta", {}).get("class_names")
    if expected_class_names is not None and saved_names is not None:
        saved_normalized = [normalize_class_name(name) for name in saved_names]
        expected_normalized = [normalize_class_name(name) for name in expected_class_names]
        if saved_normalized != expected_normalized:
            raise ValueError(
                "Failure pool class names do not match the current dataset mapping: "
                f"failures={saved_names} dataset={expected_class_names}"
            )
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


def add_placeholder_tokens(tokenizer, text_encoder, groups: dict[str, list[str]], init_mode: str,
                           initializer_token: str | None = None,
                           class_names_by_group: dict[str, str] | None = None) -> dict[str, list[int]]:
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
        elif init_mode == "class_name":
            if not class_names_by_group:
                raise ValueError("init_mode=class_name requires class_names_by_group.")
            for group, tokens in groups.items():
                class_name = class_names_by_group.get(group)
                if not class_name:
                    raise ValueError(f"missing class name for prompt group: {group}")
                init_ids = tokenizer.encode(class_name, add_special_tokens=False)
                if not init_ids:
                    raise ValueError(f"class name maps to no CLIP tokens: {class_name!r}")
                init = token_embeds[init_ids].mean(dim=0)
                for token in tokens:
                    token_embeds[tokenizer.convert_tokens_to_ids(token)] = init
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

