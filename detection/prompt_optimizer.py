from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .failures import load_failures



def token_key(class_id: int, failure_type: str, index: int = 0) -> str:
    return f"<f_{class_id}_{failure_type}_{index}>"


def prompt_key(class_id: int, failure_type: str) -> str:
    return f"{class_id}:{failure_type}"


def image_and_mask(image_path: Path, bbox, resolution: int, device):
    image = Image.open(image_path).convert("RGB")
    ow, oh = image.size
    tensor = torch.from_numpy(__import__("numpy").asarray(image.resize((resolution, resolution))).copy())
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1
    x, y, w, h = bbox
    x1, x2 = int(x / ow * resolution), int((x + w) / ow * resolution)
    y1, y2 = int(y / oh * resolution), int((y + h) / oh * resolution)
    mask = torch.zeros((1, 1, resolution, resolution), device=device)
    mask[:, :, max(0, y1):min(resolution, y2), max(0, x1):min(resolution, x2)] = 1
    target = torch.tensor([[0, 0, (x + w / 2) / ow, (y + h / 2) / oh, w / ow, h / oh]], device=device)
    return tensor, mask, target


def roi_semantic_loss(generated, original, mask):
    # Multi-scale ROI features constrain appearance without loading another large model.
    losses = []
    for scale in (1, 4, 16):
        gen = F.avg_pool2d(generated * mask, scale, scale).flatten(1)
        ref = F.avg_pool2d(original * mask, scale, scale).flatten(1)
        losses.append(1 - F.cosine_similarity(gen, ref).mean())
    return sum(losses) / len(losses) + 0.1 * ((generated - original).abs() * mask).sum() / mask.sum().clamp_min(1)


class ObjectPromptOptimizer:
    def __init__(self, cfg: dict, detector, logger: logging.Logger | None = None):
        
        from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline

        self.cfg, self.detector = cfg, detector
        self.logger = logger or logging.getLogger("detection_closed_loop")
        pcfg = cfg["prompt_optimization"]
        self.device = torch.device(pcfg["device"])
        dtype = torch.float16 if pcfg["mixed_precision"] == "fp16" else torch.float32
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(pcfg["model"], torch_dtype=dtype,
                                                                   safety_checker=None, requires_safety_checker=False)
        
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.dtype = dtype
        for module in (self.pipe.vae, self.pipe.unet, self.pipe.text_encoder):
            module.requires_grad_(False)
        self.embedding = self.pipe.text_encoder.get_input_embeddings()
        self.embedding.requires_grad_(True)

    def _embedding_prior(self, mode: str, vocab_size: int | None = None):
        token_embeds = self.pipe.text_encoder.get_input_embeddings().weight.detach().float()
        if vocab_size is not None:
            token_embeds = token_embeds[:vocab_size]
        token_embeds = token_embeds.cpu().numpy()
        if mode == "mean_emb_cov_emb":
            return token_embeds.mean(axis=0), np.cov(token_embeds, rowvar=False)
        if mode == "mean_emb_std_emb":
            return token_embeds.mean(axis=0), token_embeds.std(axis=0)
        raise ValueError(f"Unsupported embedding prior mode: {mode}")

    def _class_name_embedding(self, class_name: str) -> torch.Tensor:
        init_ids = self.pipe.tokenizer(class_name, add_special_tokens=False).input_ids
        if not init_ids:
            raise ValueError(f"Class name produced no tokenizer ids: {class_name!r}")
        return self.embedding.weight[init_ids].detach().mean(0)

    def _init_placeholder_weights(self, groups, class_names, original_vocab_size: int):
        pcfg = self.cfg["prompt_optimization"]
        init_mode = pcfg.get("init_mode", "class_name")
        init_scale = float(pcfg.get("init_noise_scale", 1.0))
        prior = None
        if init_mode in {"mean_emb_cov_emb", "mean_emb_std_emb"}:
            prior = self._embedding_prior(init_mode, original_vocab_size)

        token_embeds = self.embedding.weight.data
        for (class_id, failure_type), group_tokens in groups.items():
            token_ids = self.pipe.tokenizer.convert_tokens_to_ids(group_tokens)
            if init_mode == "class_name":
                weights = [self._class_name_embedding(class_names[class_id]) for _ in group_tokens]
            elif init_mode == "class_name_plus_noise":
                base = self._class_name_embedding(class_names[class_id]).float().cpu().numpy()
                _, std = self._embedding_prior("mean_emb_std_emb", original_vocab_size)
                sampled = np.random.normal(loc=base, scale=std * init_scale, size=(len(group_tokens), base.shape[0]))
                weights = [torch.tensor(x, dtype=token_embeds.dtype) for x in sampled]
            elif init_mode == "mean_emb_cov_emb":
                mean, cov = prior
                sampled = np.random.multivariate_normal(mean=mean, cov=cov, size=len(group_tokens))
                weights = [torch.tensor(x, dtype=token_embeds.dtype) for x in sampled]
            elif init_mode == "mean_emb_std_emb":
                mean, std = prior
                sampled = np.random.normal(loc=mean, scale=std * init_scale, size=(len(group_tokens), mean.shape[0]))
                weights = [torch.tensor(x, dtype=token_embeds.dtype) for x in sampled]
            elif init_mode == "init_token":
                init_token = pcfg.get("init_token")
                if not init_token:
                    raise ValueError("prompt_optimization.init_token is required when init_mode=init_token")
                init_ids = self.pipe.tokenizer.encode(init_token, add_special_tokens=False)
                if len(init_ids) != 1:
                    raise ValueError("prompt_optimization.init_token must map to exactly one token")
                weights = [token_embeds[init_ids[0]].detach().clone() for _ in group_tokens]
            else:
                raise ValueError(f"Invalid prompt_optimization.init_mode: {init_mode}")

            for token_id, weight in zip(token_ids, weights):
                token_embeds[token_id] = weight.to(device=token_embeds.device, dtype=token_embeds.dtype)

    def _add_tokens(self, groups, class_names):
        tokens = []
        for class_id, failure_type in groups:
            for index in range(self.cfg["prompt_optimization"]["tokens_per_prompt"]):
                tokens.append(token_key(class_id, failure_type, index))
        original_vocab_size = len(self.pipe.tokenizer)
        added = self.pipe.tokenizer.add_tokens(tokens)
        if added != len(tokens):
            raise ValueError("Placeholder token collision; use a fresh model/tokenizer")
        self.logger.info("prompt tokens added: groups=%d tokens=%d original_vocab=%d new_vocab=%d",
                         len(groups), len(tokens), original_vocab_size, original_vocab_size + added)
        self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer))
        self.embedding = self.pipe.text_encoder.get_input_embeddings()
        self.embedding.requires_grad_(True)
        with torch.no_grad():
            self._init_placeholder_weights(groups, class_names, original_vocab_size)
        return tokens

    def _clear_inactive_optimizer_state(self, optimizer, active_ids):
        state = optimizer.state.get(self.embedding.weight)
        if not state:
            return
        keep = torch.ones(len(self.embedding.weight), dtype=torch.bool, device=self.device)
        keep[active_ids] = False
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if value is not None and value.ndim > 0 and value.shape[0] == len(keep):
                value[keep] = 0

    def optimize(self, failure_file: Path, dataset, output: Path):
        failures = load_failures(failure_file)
        grouped = defaultdict(list)
        for item in failures:
            grouped[(item["class_id"], item["failure_type"])].append(item)
        if not grouped:
            raise RuntimeError("Failure pool is empty; prompt optimization has nothing to optimize")
        self.logger.info("prompt optimization data: failure_file=%s groups=%d failures=%d",
                         failure_file, len(grouped), len(failures))
        group_tokens = {
            (c, f): [token_key(c, f, i) for i in range(self.cfg["prompt_optimization"]["tokens_per_prompt"])]
            for c, f in grouped
        }
        tokens = self._add_tokens(group_tokens, dataset.class_names)
        optimizer = torch.optim.AdamW([self.embedding.weight], lr=self.cfg["prompt_optimization"]["learning_rate"],
                                      eps=1e-6, weight_decay=0.0)
        pcfg = self.cfg["prompt_optimization"]
        history = []
        for step in range(pcfg["steps"]):
            before_step = self.embedding.weight.detach().clone()
            group = random.choice(list(grouped))
            sample = random.choice(grouped[group])
            original, mask, target = image_and_mask(dataset.image_path(sample["image_id"]), sample["bbox"], pcfg["resolution"], self.device)
            tokens_for_group = " ".join(group_tokens[group])
            active_ids = [self.pipe.tokenizer.convert_tokens_to_ids(x) for x in group_tokens[group]]
            prompt = f"a photo of {tokens_for_group} {dataset.class_names[group[0]]}"
            ids = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
                                      truncation=True, return_tensors="pt").input_ids.to(self.device)
            embeds = self.pipe.text_encoder(ids)[0]
            with torch.no_grad():
                latents = self.pipe.vae.encode(original.to(self.dtype)).latent_dist.sample() * self.pipe.vae.config.scaling_factor
                masked = original * (mask < .5)
                masked_latents = self.pipe.vae.encode(masked.to(self.dtype)).latent_dist.sample() * self.pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timestep = torch.randint(pcfg["min_timestep"], pcfg["max_timestep"], (1,), device=self.device).long()
                noisy = self.pipe.scheduler.add_noise(latents, noise, timestep)
            latent_mask = F.interpolate(mask, size=latents.shape[-2:]).to(self.dtype)
            model_input = torch.cat((noisy, latent_mask, masked_latents), dim=1)
            noise_pred = self.pipe.unet(model_input, timestep, encoder_hidden_states=embeds).sample
            alpha = self.pipe.scheduler.alphas_cumprod.to(self.device)[timestep].view(1, 1, 1, 1)
            pred_x0 = (noisy - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
            generated = self.pipe.vae.decode(pred_x0.to(self.dtype) / self.pipe.vae.config.scaling_factor).sample.float().clamp(-1, 1)
            composite = generated * mask + original * (1 - mask)
            semantic = roi_semantic_loss(composite, original, mask)
            detector_image = (composite + 1) / 2
            detector_loss = self.detector.differentiable_loss(detector_image, target)
            loss = pcfg["semantic_weight"] * semantic - pcfg["detector_weight"] * detector_loss
            if not torch.isfinite(loss):
                history.append({"step": step, "group": list(group), "skipped_non_finite": True})
                self.logger.warning("prompt step skipped: step=%d/%d group=%s reason=non_finite_loss",
                                    step + 1, pcfg["steps"], prompt_key(group[0], group[1]))
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                keep = torch.ones(len(self.embedding.weight), dtype=torch.bool, device=self.device)
                keep[active_ids] = False
                self.embedding.weight.grad[keep] = 0
                self.embedding.weight.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                active_grad = self.embedding.weight.grad[active_ids]
                norm = active_grad.norm().clamp_min(1e-6)
                if norm > pcfg.get("max_grad_norm", 1.0):
                    self.embedding.weight.grad[active_ids] *= pcfg.get("max_grad_norm", 1.0) / norm
                self._clear_inactive_optimizer_state(optimizer, active_ids)
            optimizer.step()
            with torch.no_grad():
                self.embedding.weight[keep] = before_step[keep]
                if not torch.isfinite(self.embedding.weight[active_ids]).all():
                    self.embedding.weight[active_ids] = before_step[active_ids]
            history.append({"step": step, "group": list(group), "loss": float(loss),
                            "semantic": float(semantic), "detector": float(detector_loss)})
            if step == 0 or (step + 1) % pcfg.get("log_every", 10) == 0 or step + 1 == pcfg["steps"]:
                self.logger.info("prompt step: step=%d/%d group=%s loss=%.6f semantic=%.6f detector=%.6f",
                                 step + 1, pcfg["steps"], prompt_key(group[0], group[1]),
                                 float(loss), float(semantic), float(detector_loss))
        output.parent.mkdir(parents=True, exist_ok=True)
        learned = {token: self.embedding.weight[self.pipe.tokenizer.convert_tokens_to_ids(token)].detach().cpu() for token in tokens}
        torch.save(learned, output)
        metadata = {
            "groups": {prompt_key(k[0], k[1]): v for k, v in group_tokens.items()},
            "token_ids": {token: int(self.pipe.tokenizer.convert_tokens_to_ids(token)) for token in tokens},
            "init_mode": pcfg.get("init_mode", "class_name"),
            "tokens_per_prompt": pcfg["tokens_per_prompt"],
            "history": history,
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        return output


def load_prompt_groups(path: Path):
    return json.loads(path.with_suffix(".json").read_text())["groups"]
