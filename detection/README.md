# YOLOv7 closed-loop hard-example generation

This task ports adversarial prompt optimization to object detection. The loop is:

1. run YOLOv7 on **real COCO-mini images only**;
2. match every GT instance against raw detector predictions and save miss,
   localization, classification, ambiguity, low-confidence and forgetting failures;
3. optimize object textual-inversion tokens keyed by `(class_id, failure_type)`;
4. use the frozen tokens for bbox-local Stable Diffusion inpainting;
5. inherit every original GT annotation, merge generated and real image lists,
   fine-tune YOLOv7, and evaluate on untouched COCO val2017;
6. repeat, while mining the next round from the real set again.

## Setup and run

```bash
bash scripts/setup_detection.sh
bash scripts/download_coco_mini.sh
bash scripts/run_closed_loop.sh
```

The default experiment is intentionally small enough for a feasibility run but
still needs an NVIDIA GPU (roughly 16 GB VRAM; reduce resolution/batch size for
smaller cards). All controls are in `configs/detection_coco_mini.yaml`.

Individual stages are restartable:

```bash
conda run -n py10 python -m detection.runner prepare --config configs/detection_coco_mini.yaml
conda run -n py10 python -m detection.runner mine --round 0 --config configs/detection_coco_mini.yaml
conda run -n py10 python -m detection.runner optimize --round 0 --config configs/detection_coco_mini.yaml
conda run -n py10 python -m detection.runner generate --round 0 --config configs/detection_coco_mini.yaml
conda run -n py10 python -m detection.runner finetune --round 0 --config configs/detection_coco_mini.yaml
```

Prompt optimization uses a one-step `x0` estimate at a random diffusion
timestep. `L_prompt = L_semantic - lambda * L_YOLO`; the YOLOv7 training loss is
back-propagated to the image and then to placeholder embeddings. A gradient mask
and post-step restoration guarantee that no pretrained text token is changed.
The UNet, VAE, text-transformer blocks, and detector weights remain frozen.

Artifacts are written under `outputs/detection_closed_loop`: JSON failure pools,
learned token tensors, synthetic images/labels, YOLO runs, forgetting state, and
the final round summary. Synthetic manifests are tagged `source=synthetic`, and
the failure loader rejects such records to enforce real-only mining.

`detector.backend: mock` exists only for unit/smoke testing and must not be used
for reported experiments.
