# Prompt Difficulty Ablation

This experiment tests whether learned object tokens encode detector difficulty rather than only class appearance. It is independent of the round-based pipeline and does not modify existing generation, optimization, finetuning, or evaluation code.

## Controlled conditions

The three primary conditions change only the phrase attached to each GLIGEN box:

| Condition | Box phrase |
| --- | --- |
| `class` | `bird` |
| `learned` | `<det_class_14_0>,...,<det_class_14_3>` |
| `class_learned` | `bird, <det_class_14_0>,...,<det_class_14_3>` |

Use the same source images, layout boxes, filtering, seed, SDEdit strength, GLIGEN settings, and detector weights for every condition. The primary experiment uses `global_prompt_source=empty`; otherwise `category_count` leaks class names into the learned-only condition. A second experiment may use `category_count` to test learned tokens when scene-level class semantics are supplied equally to all groups.

## Outputs and interpretation

The detector analysis saves instance-level matched confidence, FN rate, class-error rate, false positives, and ROI-pooled layer-50 YOLOv7 backbone features. The report overlays detector false negatives on a t-SNE plot and computes distance to real source-image failure and detected regions.

Evidence for detector difficulty requires several signals together:

- `learned` or `class_learned` has lower matched confidence and higher FN than `class`.
- Its feature distribution is closer to the real source failure region, including within-class comparisons.
- Images still contain the intended class. A high FN rate accompanied by high class-error or obvious semantic collapse is not evidence of useful difficulty.
- Finetuning on the condition improves held-out real COCO AP/AP50, especially target-class AP, from the same initial weights and training budget.
- A source-class token transferred as `target class + source learned token` still raises difficulty on other classes if it represents a general hard pattern.

## Create commands

From the repository root:

```bash
conda run -n py10 python -m detection_gligen_sdedit.experiments.prompt_vis.prompt_difficulty_ablation.make_prompt_difficulty_commands \
  --experiment-dir outputs/prompt_ablation/detection_gligen_sdedit/prompt_difficulty_0807 \
  --pretrained-model-name-or-path /data/model/models_zwq/gligen/diffusers-generation-text-box \
  --annotations data/coco/annotations/instances_minitrain2017.json \
  --images data/coco/images/train2017 \
  --tokens-dir outputs/detection_gligen_sdedit/exp_0807_30class_renew/round_1/prompts \
  --detector-weights outputs/cocofromgr/epoch_300.pt \
  --classes-subset 30 \
  --max-images 128 \
  --global-prompt-source empty \
  --device cuda:0
```

This writes `commands.sh` and `experiment_config.json`. Execute individual sections first, or run all generation and initial analysis stages:

```bash
bash outputs/prompt_ablation/detection_gligen_sdedit/prompt_difficulty_0807/commands.sh
```

The final report is `report/index.html` under the experiment directory.

To include independent finetuning and full validation commands, add:

```text
--include-finetune
--train-annotations data/coco/annotations/instances_minitrain2017.json
--train-images data/coco/images/train2017
--val-annotations data/coco/annotations/instances_val2017.json
--val-images data/coco/images/val2017
--real-train-list outputs/detection_gligen_sdedit/exp_0807_30class_renew/datasets/real_train/images.txt
--real-val-list outputs/detection_gligen_sdedit/exp_0807_30class_renew/datasets/real_val/images.txt
--baseline-metrics outputs/detection_gligen_sdedit/exp_0807_30class_renew/round_1/evaluation/metrics.json
```

## Cross-class transfer

`generate_prompt_difficulty_condition.py` accepts `--transfer-source-class-id`. For the strongest transfer test, use `--phrase-mode class_learned`: the target class name preserves semantics while the learned tokens come from another class. Compare source-target pairs with similar object scale first, then unrelated categories. The `--transfer-source-class-ids` option in the command builder creates a quick learned-only transfer screen; promising source tokens should then be rerun with `class_learned` for a semantic-controlled test.
