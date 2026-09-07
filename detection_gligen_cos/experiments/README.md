# Prompt qualitative and ablation experiments

This folder contains helper scripts that do not change the main closed-loop pipeline.

## 1. Qualitative standalone prompt sampling

Use this to inspect what the learned tokens encode without inpainting into a real bbox.

For each class prompt group it generates three variants:

- `class_only`: `bird, photo, highly detailed, photorealistic`
- `tokens_only`: `<det_class_14_0>,..., photo, highly detailed, photorealistic`
- `class_tokens`: `bird, <det_class_14_0>,..., photo, highly detailed, photorealistic`

Example:

```bash
conda run -n py10 python -m detection_gligen_sdedit.experiments.sample_prompts \
  --pretrained-model-name-or-path /data/zwq/models/stable-diffusion-inpainting \
  --tokens-dir outputs/detection_gligen_sdedit/exp_0723/round_1/prompts \
  --checkpoint learned_embeds-500.bin \
  --output-dir outputs/detection_gligen_sdedit/prompt_ablation_0723/qual/round_1_step_500 \
  --num-images 4 \
  --seed 23 \
  --device cuda:0 \
  --local-files-only
```

Outputs:

- per-image samples under `output_dir/<checkpoint>/<group>/<variant>/`
- one `grid.jpg` per class group
- `manifest.json` with exact prompts and seeds

## 2. Step and detector_weight ablation

Generate configs and command lists:

```bash
conda run -n py10 python -m detection_gligen_sdedit.experiments.make_ablation_configs \
  --base-config detection_gligen_sdedit/configs/coco_mini_exp_0723.yaml \
  --output-root outputs/detection_gligen_sdedit/prompt_ablation_0723
```

This writes:

- `outputs/detection_gligen_sdedit/prompt_ablation_0723/configs/*.yaml`
- `outputs/detection_gligen_sdedit/prompt_ablation_0723/commands.sh`
- `outputs/detection_gligen_sdedit/prompt_ablation_0723/commands.json`

Suggested workflow:

1. Run `prompt_sampling_existing` commands first to inspect current checkpoints.
2. Run quick `steps` ablations: fixed `detector_weight=0.1`, vary `max_train_steps`.
3. Run quick `detector_weight` ablations: fixed `steps=500`, vary `detector_weight`.
4. Pick 2-3 representative configs for full `optimize -> generate -> finetune -> evaluate`.

