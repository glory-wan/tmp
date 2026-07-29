from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from detection.detector import YoloV7Detector

from .common import (
    CocoMini,
    add_placeholder_tokens,
    load_depth_style_components,
    load_failures,
    mask_tensor_from_bbox,
    patch_huggingface_hub_for_diffusers_024,
    parse_class_subset,
    placeholder_tokens,
    prompt_group_key,
    roi_semantic_loss,
    setup_logger,
    subset_label,
    tensor_from_image,
)


def save_group_embeddings(text_encoder, tokenizer, groups, output_dir: Path, step: int, metadata: dict) -> Path:
    learned = {}
    for group, tokens in groups.items():
        for token in tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            learned[token] = text_encoder.get_input_embeddings().weight[token_id].detach().cpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"learned_embeds-{step}.bin"
    torch.save(learned, out)
    (output_dir / "object_prompts.json").write_text(json.dumps({**metadata, "groups": groups, "latest": out.name}, indent=2), encoding="utf-8")
    return out


def load_resume_embeddings(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"resume token file not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"resume token file must contain a token->embedding dict: {path}")
    return data


def load_resume_prompt_groups(path: str | Path) -> dict[str, list[str]]:
    """Load prompt groups saved next to a learned_embeds-*.bin checkpoint."""
    meta_path = Path(path).parent / "object_prompts.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    groups = meta.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError(f"invalid groups field in resume metadata: {meta_path}")
    return {str(group): [str(token) for token in tokens] for group, tokens in groups.items()}


def merge_resume_prompt_groups(current_groups: dict[str, list[str]],
                               resume_groups: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[str]]:
    """Keep previous groups, including groups without current-round failures."""
    merged = {group: list(tokens) for group, tokens in resume_groups.items()}
    for group, tokens in current_groups.items():
        merged.setdefault(group, list(tokens))
    preserved = sorted(set(resume_groups) - set(current_groups))
    return {group: merged[group] for group in sorted(merged)}, preserved


def apply_resume_embeddings(text_encoder, tokenizer, groups: dict[str, list[str]], resume_embeddings: dict[str, torch.Tensor]) -> tuple[int, list[str]]:
    token_embeds = text_encoder.get_input_embeddings().weight.data
    expected_tokens = [token for tokens in groups.values() for token in tokens]
    loaded = 0
    missing = []
    for token in expected_tokens:
        if token not in resume_embeddings:
            missing.append(token)
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        token_embeds[token_id] = resume_embeddings[token].to(device=token_embeds.device, dtype=token_embeds.dtype)
        loaded += 1
    return loaded, missing


def active_target(sample, original_size, device):
    ow, oh = original_size
    x, y, w, h = sample["bbox"]
    return torch.tensor([[0, sample["class_id"], (x + w / 2) / ow, (y + h / 2) / oh, w / ow, h / oh]], device=device)


def build_object_prompt(class_name: str, tokens: list[str], include_class_name: bool) -> str:
    token_text = ",".join(tokens)
    if include_class_name:
        return f"{class_name}, {token_text}, photo, highly detailed, photorealistic"
    return f"{token_text}, photo, highly detailed, photorealistic"


def main():
    parser = argparse.ArgumentParser(description="Depth-style object prompt optimization for YOLOv7 failures.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--yolov7", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-scope", choices=["class", "class_failure"], default="class")
    parser.add_argument("--include-class-name", action="store_true",
                        help="Prepend the explicit COCO class name to the learned-token object prompt.")
    parser.add_argument("--num-new-tokens", type=int, default=4)
    parser.add_argument("--initializer-token", default="object")
    parser.add_argument("--init-mode", choices=["initializer_token", "mean_emb_cov_emb"], default="mean_emb_cov_emb")
    parser.add_argument("--resume-token", default=None,
                        help="Path to a previous learned_embeds-*.bin file for continuing object prompt optimization.")
    parser.add_argument("--resume-mode", choices=["overwrite", "reinit"], default="overwrite",
                        help="When --resume-token is set: overwrite matching placeholder embeddings with the checkpoint, or keep normal reinitialization.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--mixed-precision", default="fp16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--detector-weight", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--classes-subset", type=int, default=None)
    parser.add_argument("--class-ids", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger(args.output_dir)
    accelerator = Accelerator(mixed_precision=args.mixed_precision, project_config=ProjectConfiguration(project_dir=args.output_dir))
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    patch_huggingface_hub_for_diffusers_024()
    from diffusers.optimization import get_scheduler

    class_subset = parse_class_subset(args.classes_subset is not None or args.class_ids is not None, args.classes_subset, args.class_ids)
    dataset = CocoMini(args.annotations, args.images, class_subset=class_subset)
    logger.info("dataset loaded: images=%d annotations=%d classes=%d",
                len(dataset.images), sum(len(v) for v in dataset.annotations.values()), len(dataset.class_names))
    logger.info("class subset: %s", subset_label(class_subset))
    failures = load_failures(args.failures, class_subset=class_subset)
    grouped = defaultdict(list)
    for item in failures:
        grouped[prompt_group_key(item["class_id"], item["failure_type"], args.prompt_scope)].append(item)
    if not grouped:
        raise RuntimeError("Empty failure pool.")
    current_groups = {group: placeholder_tokens(group, args.num_new_tokens) for group in sorted(grouped)}
    resume_source = None
    resume_groups: dict[str, list[str]] = {}
    preserved_resume_groups: list[str] = []
    if args.resume_token:
        resume_source = str(Path(args.resume_token))
        if args.resume_mode == "overwrite":
            resume_groups = load_resume_prompt_groups(args.resume_token)
            if not resume_groups:
                logger.warning("resume metadata object_prompts.json not found or contains no groups next to %s; only current failure groups will be saved",
                               args.resume_token)
            groups, preserved_resume_groups = merge_resume_prompt_groups(current_groups, resume_groups)
        else:
            groups = current_groups
    else:
        groups = current_groups
    group_sizes = {group: len(items) for group, items in grouped.items()}
    if args.resume_token and args.resume_mode == "overwrite":
        logger.info("resume prompt groups merged: resume_groups=%d train_groups=%d preserved_without_current_failures=%d",
                    len(resume_groups), len(current_groups), len(preserved_resume_groups))
        if preserved_resume_groups:
            logger.info("preserved resume groups without current failures: %s",
                        preserved_resume_groups[:30] + (["..."] if len(preserved_resume_groups) > 30 else []))
    logger.info("optimize start: failures=%d train_groups=%d saved_groups=%d scope=%s group_sizes=%s",
                len(failures), len(current_groups), len(groups), args.prompt_scope, group_sizes)

    logger.info("loading SD components: model=%s local_files_only=%s dtype=%s",
                args.pretrained_model_name_or_path, args.local_files_only, weight_dtype)
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_depth_style_components(
        args.pretrained_model_name_or_path,
        dtype=weight_dtype,
        local_files_only=args.local_files_only,
        text_encoder_dtype=torch.float32,
    )
    group_token_ids = add_placeholder_tokens(tokenizer, text_encoder, groups, args.init_mode, args.initializer_token)
    all_added_ids = [idx for ids in group_token_ids.values() for idx in ids]
    resume_loaded = 0
    resume_missing: list[str] = []
    if args.resume_token:
        if args.resume_mode == "overwrite":
            resume_embeddings = load_resume_embeddings(args.resume_token)
            resume_loaded, resume_missing = apply_resume_embeddings(text_encoder, tokenizer, groups, resume_embeddings)
            logger.info("resume token applied: file=%s mode=%s loaded=%d missing=%d",
                        args.resume_token, args.resume_mode, resume_loaded, len(resume_missing))
            if resume_missing:
                logger.info("resume token missing placeholders: %s", resume_missing[:20] + (["..."] if len(resume_missing) > 20 else []))
        else:
            logger.info("resume token ignored by mode=reinit: file=%s; using init_mode=%s", args.resume_token, args.init_mode)
    logger.info("text encoder embedding dtype: %s", text_encoder.get_input_embeddings().weight.dtype)
    if text_encoder.get_input_embeddings().weight.dtype != torch.float32:
        raise RuntimeError("Textual inversion embeddings must stay FP32 during optimization, matching the depth implementation.")
    logger.info("placeholder tokens ready: total_tokens=%d init_mode=%s token_id_range=%s-%s",
                len(all_added_ids), args.init_mode, min(all_added_ids), max(all_added_ids))

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    detector = YoloV7Detector(args.yolov7, args.weights, "0", 640)
    logger.info("detector loaded: yolov7=%s weights=%s", args.yolov7, args.weights)
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps)
    text_encoder, optimizer, lr_scheduler = accelerator.prepare(text_encoder, optimizer, lr_scheduler)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    detector.model.to(accelerator.device)
    noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
    timesteps = noise_scheduler.timesteps
    t_index = max(args.num_inference_steps - min(int(args.num_inference_steps * args.strength), args.num_inference_steps), 0)
    timesteps = timesteps[t_index:]
    orig_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    all_added_ids_tensor = torch.tensor(all_added_ids, dtype=torch.long, device=accelerator.device)
    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    logger.info("training start: steps=%d lr=%s scheduler=%s strength=%s denoise_steps=%d semantic_weight=%s detector_weight=%s",
                args.max_train_steps, args.learning_rate, args.lr_scheduler, args.strength, len(timesteps),
                args.semantic_weight, args.detector_weight)
    logger.info("detector loss target policy: full composite image input; single failure-instance target only")
    logger.info("prompt format: include_class_name=%s template='%s'",
                args.include_class_name,
                "{class_name}, {tokens}, photo, highly detailed, photorealistic" if args.include_class_name
                else "{tokens}, photo, highly detailed, photorealistic")

    for global_step in progress:
        group = random.choice(list(grouped))
        sample = random.choice(grouped[group])
        image_pil, original = tensor_from_image(dataset.image_path(sample["image_id"]), args.resolution, accelerator.device, torch.float32)
        mask = mask_tensor_from_bbox(image_pil.size, sample["bbox"], args.resolution, accelerator.device, torch.float32)
        target = active_target(sample, image_pil.size, accelerator.device)
        class_name = dataset.class_names[int(sample["class_id"])]
        prompt = build_object_prompt(class_name, groups[group], args.include_class_name)
        input_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(accelerator.device)
        prompt_embeds = text_encoder(input_ids)[0].to(dtype=weight_dtype)

        with accelerator.accumulate(text_encoder):
            latents = vae.encode(original.to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            masked_image = original * (mask < 0.5)
            masked_latents = vae.encode(masked_image.to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timestep = timesteps[torch.randint(0, len(timesteps), (1,), device=accelerator.device)]
            noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)
            latent_mask = F.interpolate(mask, size=latents.shape[-2:]).to(dtype=weight_dtype)
            model_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)
            model_input = noise_scheduler.scale_model_input(model_input, timestep)
            noise_pred = unet(model_input, timestep, encoder_hidden_states=prompt_embeds).sample
            alpha = noise_scheduler.alphas_cumprod.to(accelerator.device)[timestep].view(1, 1, 1, 1)
            pred_x0 = (noisy_latents - (1 - alpha).sqrt() * noise_pred) / alpha.sqrt()
            decoded = vae.decode(pred_x0.to(dtype=weight_dtype) / vae.config.scaling_factor).sample.float().clamp(-1, 1)
            composite = decoded * mask + original * (1 - mask)
            semantic = roi_semantic_loss(composite, original, mask)
            detector_input = (composite + 1) / 2
            detector_target = target
            detector_loss = detector.differentiable_loss(detector_input, detector_target)
            
            loss = args.semantic_weight * semantic - args.detector_weight * detector_loss
            
            accelerator.backward(loss)
            active_ids = group_token_ids[group]
            inactive_added_ids = [idx for idx in all_added_ids if idx not in set(active_ids)]
            weights_before_step = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data
            inactive_added_before = (
                weights_before_step[inactive_added_ids].detach().clone()
                if inactive_added_ids else None
            )
            if accelerator.sync_gradients:
                accelerator.unscale_gradients(optimizer)
                grad = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.grad
                keep = torch.ones(grad.shape[0], dtype=torch.bool, device=grad.device)
                keep[active_ids] = False
                grad[keep] = 0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            with torch.no_grad():
                weights = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
                # Freeze the original CLIP vocabulary, but keep every learned
                # placeholder token across steps.  Earlier code restored every
                # non-active token to orig_embeds, which erased other classes'
                # learned prompts whenever a different class was sampled.
                restore_original_vocab = torch.ones(weights.shape[0], dtype=torch.bool, device=weights.device)
                restore_original_vocab[all_added_ids_tensor] = False
                weights[restore_original_vocab] = orig_embeds[restore_original_vocab]
                if inactive_added_ids:
                    weights[inactive_added_ids] = inactive_added_before.to(device=weights.device, dtype=weights.dtype)

        if global_step == 0 or (global_step + 1) % args.log_every == 0:
            progress.set_postfix(loss=float(loss), semantic=float(semantic), det=float(detector_loss), group=group)
            logger.info("step=%d/%d group=%s loss=%.6f semantic=%.6f det=%.6f lr=%.6g",
                        global_step + 1, args.max_train_steps, group, float(loss), float(semantic), float(detector_loss), lr_scheduler.get_last_lr()[0])
        if accelerator.is_main_process and ((global_step + 1) % args.save_steps == 0 or global_step + 1 == args.max_train_steps):
            metadata = {"prompt_scope": args.prompt_scope, "num_new_tokens": args.num_new_tokens,
                        "init_mode": args.init_mode,
                        "resume_token": resume_source,
                        "resume_mode": args.resume_mode,
                        "resume_loaded": resume_loaded,
                        "resume_missing": resume_missing,
                        "train_groups": sorted(grouped),
                        "preserved_resume_groups": preserved_resume_groups,
                        "include_class_name": args.include_class_name,
                        "class_subset": sorted(class_subset) if class_subset is not None else None}
            out = save_group_embeddings(accelerator.unwrap_model(text_encoder), tokenizer, groups, Path(args.output_dir), global_step + 1, metadata)
            logger.info("saved embeddings: %s", out)
        progress.update(1)
    accelerator.wait_for_everyone()
    logger.info("training complete: output_dir=%s latest_metadata=%s", args.output_dir, Path(args.output_dir) / "object_prompts.json")


if __name__ == "__main__":
    main()
