# Prompt 梯度对齐闭环使用说明

## 快速启动

```bash
cd /home/suhu/data/wgr/code/prompt_stepbys
python -m detection_gligen_cos.retrain_runner \
  --overall-yaml detection_gligen_cos/configs/coco_bus_bird_umbrella_retrain.yaml
```

中断后使用 schema 2 状态文件恢复：

```bash
python -m detection_gligen_cos.retrain_runner \
  --state-json outputs/bus_bird_umbrella_gligen_cos_prompt_retrain_loop/retrain_state.json
```

V1 的 schema 1 状态文件不兼容本版本，应使用新的 workspace 启动实验。

## 闭环流程

```text
baseline / 上一轮 cos_prompt best.pt
  → mine
  → Prompt optimization + Guide gradient cosine loss
  → full-image GLIGEN-SDEdit generation
  → fresh retrain: train2017 + 第 1..当前轮全部 synthetic
  → 下一轮
```

生成后不再调用 `build_coco_cos_subsets_ultralytics.py`，不创建 cos 正、负、零
子集，每轮只训练一个检测器。根目录脚本仍可用于离线评分和数值审计。

## 命令行参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--overall-yaml` | `configs/coco_bus_bird_umbrella_retrain.yaml` | 新任务的完整配置。 |
| `--state-json` | `null` | 恢复已有状态；若路径不存在，也可指定新任务状态输出位置。 |
| `--baseline` | YAML 中的 `baseline` | 覆盖第一轮检测器 checkpoint。 |
| `--model-yaml` | YAML 中的 `model_yaml` | 覆盖 retrain 从头初始化使用的模型结构。 |
| `--dataset` | YAML 中的 `dataset` | 覆盖原始数据集 YAML。 |
| `--round` | YAML 中的 `rounds` | 覆盖闭环轮数。 |
| `--syn-smaple` / `--syn-sample` | YAML 中的 `syn_sample` | 覆盖每轮合成样本数。 |
| `--device` | YAML 中的 `device` | 单个物理 GPU ID或 `cpu`。 |

## Prompt 梯度对齐

配置位于 `prompt_optimization.gradient_alignment`：

```yaml
gradient_alignment:
  enabled: true
  guide_root: null
  guide_split: val
  task: detect
  parameter_scope: detection_head
  image_size: 640
  scaleup: false
  guide_batch_size: 8
  workers: 4
  target_scope: full_layout
  loss: hinge
  weight: 1.0
  margin: 0.0
  warmup_steps: 100
  epsilon: 1.0e-12
  implementation: exact
  cache: true
  max_objects_per_image: 20
  enable_layout_filter: false
  min_bbox_area_ratio: 0.001
```

每轮先在 Guide 验证集上计算检测头平均梯度 `g_guide`。Prompt step 对当前
可微合成图像计算 `g_syn`，加入：

```text
L_cos = ReLU(margin - cosine(g_syn, g_guide))

L = semantic_weight * L_semantic
  + class_semantic_weight * L_class_semantic
  - detector_weight * L_detector
  + effective_cos_weight * L_cos
```

其中原有 `L_detector` 仍使用单个 failure target；`L_cos` 使用 failure 优先的
完整 layout target。`effective_cos_weight` 在 `warmup_steps` 内从 0 线性增加到
`weight`。实现使用 `create_graph=True` 的精确二阶梯度，不会静默降级。

Guide 图像默认来自数据集根目录的 `images/val2017`，标签来自
`labels/val2017`；YOLO 分割多边形标签会按旧筛选脚本的规则转换为外接框。
每轮缓存位于：

```text
round_<t>/prompts/gradient_alignment/guide_gradients.pt
round_<t>/prompts/gradient_alignment/guide_metadata.json
```

checkpoint、Guide 内容、参数顺序或预处理配置变化后缓存会失效并重新计算。
Prompt 日志和 `object_prompts.json` 会记录 cos、cos loss、dot、两侧梯度范数、
正值比例及 Guide metadata hash。

## 单分支 retrain

第 t 轮生成 `round_<t>/datasets/cos_prompt.yaml`，训练集包含：

```text
原始 images/train2017
<workspace>/round_1/synthetic/images
...
<workspace>/round_t/synthetic/images
```

合成目录使用绝对路径，不复制回原始数据集。每轮从 `model_yaml` 新建模型，
默认 run name 为 `round_<t>_cos_prompt`。下一轮和最终输出都使用该 run 的
`weights/best.pt`。

## 恢复规则

| 阶段 | 恢复行为 |
| --- | --- |
| mine | 重新执行当前轮挖掘。 |
| optimize | 从最新 `learned_embeds-<step>.bin` 恢复；metadata 一致时复用 Guide cache。 |
| generate | 从原子保存的 `manifest.json` 继续未完成图像。 |
| retrain | 从单一 run 的 `weights/last.pt` 调用 Ultralytics `resume=True`。 |

状态 JSON 中每轮只有 `mine`、`optimize`、`generate`、`retrain`，不存在
`align` 或 `retrain.groups`。

## 首次运行建议

精确余弦包含二阶反传，完整实验前建议把 Guide、failure、Prompt step、生成数和
retrain epoch 缩到最小规模，确认 token 梯度有限且非零、检测器权重未变化、
显存峰值可接受后再启动完整闭环。
