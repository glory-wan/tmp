# Prompt Visualization Modes

This directory contains instance-aware prompt visualization tools.

The existing `detection_gligen_sdedit.experiments.sample_prompts` script is kept as
a standalone token sanity check. It uses plain text-to-image sampling and does
not validate bbox-level prompt optimization behavior.

## ROI Inpainting

Primary visualization for the current generation method. It applies learned
tokens to failure-instance bbox masks on the original image and writes an HTML
grid with original image, bbox/mask overlay, inpaint result, prompt, class info,
and optional before/after detector scores.

```bash
conda run -n py10 python -m detection_gligen_sdedit.experiments.prompt_vis.sample_prompt_roi_inpaint \
  --pretrained-model-name-or-path /data/model/models_zwq/stable-diffusion-inpainting \
  --annotations data/coco/annotations/instances_minitrain2017.json \
  --images data/coco/images/train2017 \
  --failures outputs/detection_gligen_sdedit/exp_0728_loss/round_1/failures.json \
  --tokens-dir outputs/detection_gligen_sdedit/exp_0728_loss/round_1/prompts \
  --checkpoint learned_embeds-1000.bin \
  --output-dir outputs/prompt_vis/roi_inpaint/round_1_step_1000 \
  --max-instances 64 \
  --class-ids 14,25,53 \
  --device cuda:0 \
  --local-files-only
```

Add detector scoring when needed:

```bash
  --yolov7 external/yolov7 \
  --weights outputs/cocofromgr/epoch_300.pt \
  --detector-device 0
```

## GLIGEN Layout

Secondary visualization for layout-conditioned generation. It keeps learned
tokens unchanged, groups failure instances by source image, passes token prompts
and normalized boxes to GLIGEN, and writes an HTML grid for comparison.

```bash
conda run -n py10 python -m detection_gligen_sdedit.experiments.prompt_vis.sample_prompt_gligen \
  --pretrained-model-name-or-path /data/model/models_zwq/gligen/diffusers-generation-text-box \
  --variant fp16 \
  --annotations data/coco/annotations/instances_minitrain2017.json \
  --images data/coco/images/train2017 \
  --failures outputs/detection_gligen_sdedit/exp_0728_loss/round_1/failures.json \
  --tokens-dir outputs/detection_gligen_sdedit/exp_0728_loss/round_1/prompts \
  --checkpoint learned_embeds-1000.bin \
  --output-dir outputs/prompt_vis/gligen/round_1_step_1000 \
  --max-scenes 32 \
  --max-objects-per-scene 6 \
  --class-ids 14,25,53 \
  --device cuda:0 \
  --local-files-only
```
