from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw



def load_prompt_groups(path: Path):
    return json.loads(path.with_suffix(".json").read_text())["groups"]


class HardExampleGenerator:
    def __init__(self, cfg: dict, embeddings: Path):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        gcfg = cfg["generation"]
        dtype = torch.float16 if gcfg["mixed_precision"] == "fp16" else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(gcfg["model"], torch_dtype=dtype,
                                                                   safety_checker=None, requires_safety_checker=False)
       
        # learned = torch.load(embeddings, map_location="cpu")
        # for token, embedding in learned.items():
        #     self.pipe.load_textual_inversion({token: embedding}, token=token)
        # self.pipe.to(gcfg["device"])

        learned = torch.load(embeddings, map_location="cpu")

        tokens = list(learned.keys())
        added = self.pipe.tokenizer.add_tokens(tokens)

        if added != len(tokens):
            existing = len(tokens) - added
            raise ValueError(
                f"Textual-inversion token collision: "
                f"requested={len(tokens)}, added={added}, existing={existing}"
            )

        self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer))
        input_embeddings = self.pipe.text_encoder.get_input_embeddings()

        with torch.no_grad():
            for token, embedding in learned.items():
                token_id = self.pipe.tokenizer.convert_tokens_to_ids(token)
                input_embeddings.weight[token_id].copy_(
                    embedding.to(
                        device=input_embeddings.weight.device,
                        dtype=input_embeddings.weight.dtype,
                    )
                )
        print(f"[generation] loaded {len(tokens)} learned tokens")

        self.pipe.to(gcfg["device"])
        
        print(
            "[generation] pipeline devices:",
            f"unet={next(self.pipe.unet.parameters()).device}",
            f"vae={next(self.pipe.vae.parameters()).device}",
            f"text_encoder={next(self.pipe.text_encoder.parameters()).device}",
        )

        

        self.cfg, self.groups = cfg, load_prompt_groups(embeddings)

    @staticmethod
    def mask(image: Image.Image, bbox, padding: float):
        x, y, w, h = bbox
        px, py = w * padding, h * padding
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((max(0, x - px), max(0, y - py), min(image.width, x + w + px),
                                        min(image.height, y + h + py)), fill=255)
        return mask

    def generate(self, dataset, failures: list[dict], output: Path, round_index: int,
                 logger: logging.Logger | None = None):
        import torch
        gcfg = self.cfg["generation"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "images").mkdir(exist_ok=True)
        (output / "labels").mkdir(exist_ok=True)
        # Generation templates are real images selected by their mined real failures.
        by_image = {}
        for failure in failures:
            by_image.setdefault(int(failure["image_id"]), []).append(failure)
        manifest = []
        generator = torch.Generator(device=gcfg["device"]).manual_seed(self.cfg["seed"] + round_index)
        for image_id, active in list(by_image.items())[:gcfg["max_images"]]:
            image = Image.open(dataset.image_path(image_id)).convert("RGB")
            edited = 0
            for failure in active[:gcfg["max_objects_per_image"]]:
                key = f"{failure['class_id']}:{failure['failure_type']}"
                tokens = self.groups.get(key)
                if not tokens:
                    continue
                prompt = f"a realistic photo of {' '.join(tokens)} {dataset.class_names[failure['class_id']]}"
                result = self.pipe(prompt=prompt, negative_prompt=gcfg["negative_prompt"], image=image,
                                   mask_image=self.mask(image, failure["bbox"], gcfg["mask_padding"]),
                                   strength=gcfg["strength"], guidance_scale=gcfg["guidance_scale"],
                                   num_inference_steps=gcfg["inference_steps"], generator=generator).images[0]
                image = result.resize(image.size)
                edited += 1
            name = f"r{round_index}_{image_id:012d}.jpg"
            image.save(output / "images" / name, quality=95)
            info = dataset.images[image_id]
            lines = []
            for ann in dataset.annotations.get(image_id, []):
                x, y, w, h = ann["bbox"]
                lines.append(f"{ann['class_id']} {(x+w/2)/info['width']:.8f} {(y+h/2)/info['height']:.8f} "
                             f"{w/info['width']:.8f} {h/info['height']:.8f}")
            (output / "labels" / Path(name).with_suffix(".txt")).write_text("\n".join(lines) + "\n")
            manifest.append({"image_id": image_id, "file_name": name, "source": "synthetic", "round": round_index,
                             "labels_inherited_from_real": True})
            if logger and (len(manifest) == 1 or len(manifest) % 25 == 0 or len(manifest) == min(len(by_image), gcfg["max_images"])):
                logger.info("generation progress: images=%d/%d latest=%s edited_objects=%d",
                            len(manifest), min(len(by_image), gcfg["max_images"]), name, edited)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def merge_image_lists(real_list: Path, synthetic_dir: Path, output: Path) -> Path:
    real = [x for x in real_list.read_text().splitlines() if x]
    synthetic = sorted(str(x.resolve()) for x in (synthetic_dir / "images").glob("*.jpg"))
    output.write_text("\n".join(real + synthetic) + "\n")
    return output
