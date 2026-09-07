# Ultralytics 梯度匹配脚本使用说明

对应脚本：`build_coco_cos_subsets_ultralytics.py`

该脚本先计算 Guide set 的平均检测损失梯度，再逐张计算 source 图像的梯度，
并按二者的全局余弦相似度把 source 划分为正、负和零值三个子集。
```text
python /home/suhu/data/wgr/code/prompt_stepbys/build_coco_cos_subsets_ultralytics.py 
--source-root /home/suhu/data/wgr/code/prompt_stepbys/outputs/bus_bird_umbrella2/round_1/synthetic 
--guide-root /home/suhu/data/wgr/datasets/coco_bus_bird_umbrella 
--model /home/suhu/data/wgr/code/prompt_stepbys/ultralytics/runs/detect/yolo26n_COCO_subset/round_1_cos_more_0_bus_bird_umbrella/weights/best.pt 
--output-root /home/suhu/data/wgr/datasets/coco_bus_bird_umbrella 
--positive-name round_2_train_cos_more_0 
--negative-name round_2_train_cos_less_0
```

## Source 与 Guide 输入

现在使用两个相互独立的必填参数：

- `--source-root`：需要被梯度对齐和划分的图像、标签根目录。
- `--guide-root`：用于计算平均参考梯度的 Guide set 根目录。

脚本会按以下顺序自动识别目录：

| 参数 | 自动识别的目录结构 |
|---|---|
| `--source-root` | 优先使用 `images/` 和 `labels/`；其次尝试 `images/train2017` 和 `labels/train2017`；最后尝试 `images/minitrain2017` 和 `labels/minitrain2017`。 |
| `--guide-root` | 优先使用 `images/val2017` 和 `labels/val2017`；其次尝试 `images/` 和 `labels/`。 |

例如，下面两个输入会被解析为：

```text
source:
  /home/suhu/data/wgr/code/prompt_stepbys/outputs/coco_bus_dog_umbrella/round_1/synthetic/images
  /home/suhu/data/wgr/code/prompt_stepbys/outputs/coco_bus_dog_umbrella/round_1/synthetic/labels

Guide:
  /home/suhu/data/wgr/datasets/coco_bus_dog_umbrella/images/val2017
  /home/suhu/data/wgr/datasets/coco_bus_dog_umbrella/labels/val2017
```

脚本不再要求 source 或 Guide 必须具有固定图像数量，因此
`--expected-train-images` 和 `--expected-guide-images` 已删除。输入目录必须非空，
且每张图像必须存在同名 `.txt` 标签。

检测框五列标签可以直接使用。若标签是 YOLO 分割多边形格式，脚本会在内存中将其
转换为外接检测框，用于计算检测损失；不会修改原始标签文件。

## 参数列表

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--model` / `--weights` | `ultralytics/yolo26n.pt` | 指定 Ultralytics 模型权重或 YAML 配置。 |
| `--task` | `detect` | 模型任务；可选 `auto`、`detect`、`segment`、`pose`、`obb`。使用 `auto` 时从模型推断任务。 |
| `--ultralytics-root` | 当前目录下的 `ultralytics` | 本地 Ultralytics 源码目录。 |
| `--source-root` | 必填 | 需要计算梯度余弦并划分子集的 source 根目录。 |
| `--guide-root` | 必填 | Guide set 根目录，通常传入包含验证集的完整数据集根目录。 |
| `--output-root` | `/home/suhu/data/wgr/datasets/cocomini_cos_ultralytics/模型名` | 正、负、零值子集和汇总文件的输出目录。 |
| `--score-dir` | `输出目录/gradient_scores_ultralytics_val` | Guide 梯度、source 余弦分数和缓存目录。 |
| `--positive-name` | `train_cos_more_0` | 余弦大于 0 子集的目录名。 |
| `--negative-name` | `train_cos_less_0` | 余弦小于 0 子集的目录名。 |
| `--zero-name` | `train_cos_equal_0` | 余弦等于 0 子集的目录名。 |
| `--zero-policy` | `separate` | 余弦等于 0 的图像单独存放；目前唯一可选值为 `separate`。 |
| `--imgsz` | `640` | 模型输入图像尺寸。 |
| `--scaleup` / `--no-scaleup` | `--no-scaleup` | 是否允许 letterbox 预处理放大小图像。 |
| `--cuda-visible-devices` | `6` | 指定可见的物理 GPU 编号。 |
| `--device` | `0` | 程序内部使用的逻辑 GPU 编号，也可设为 `cpu`。物理 GPU 6 被设为唯一可见设备后，其逻辑编号是 0。 |
| `--guide-batch-size` | `8` | 计算 Guide 平均梯度时的 batch size。 |
| `--workers` | `4` | 数据加载进程数量。 |
| `--copy-workers` | `8` | 复制正、负、零值子集文件时的并行线程数量。 |
| `--max-source-images` / `--max-train-images` | 不限制 | 只处理按文件名排序后的前 N 张 source 图像，主要用于小规模测试；后者是兼容旧命令的别名。 |
| `--max-guide-images` | 不限制 | 只使用按文件名排序后的前 N 张 Guide 图像，主要用于小规模测试。 |
| `--resume` / `--no-resume` | `--resume` | 是否复用已有 Guide 梯度和 source 评分。输入或配置变化时应使用新的输出目录，或传入 `--no-resume`。 |
| `--skip-gradient-scoring` | 关闭 | 跳过梯度计算，直接使用已有的完整 `scores.jsonl` 重新划分子集。 |

## 当前 round 的完整命令

```bash
cd /home/suhu/data/wgr/code/prompt_stepbys

/data/users/suhu/conda/envs/wgr-G/bin/python build_coco_cos_subsets_ultralytics.py \
  --model /home/suhu/data/wgr/code/prompt_stepbys/ultralytics/detect/yolo26n_COCO_subset/bus_dog_umbrella/weights/best.pt \
  --ultralytics-root /home/suhu/data/wgr/code/prompt_stepbys/ultralytics \
  --source-root /home/suhu/data/wgr/code/prompt_stepbys/outputs/coco_bus_dog_umbrella/round_1/synthetic \
  --guide-root /home/suhu/data/wgr/datasets/coco_bus_dog_umbrella \
  --output-root /home/suhu/data/wgr/code/prompt_stepbys/outputs/coco_bus_dog_umbrella/round_1/gradient_cos_subsets \
  --cuda-visible-devices 6 \
  --device 0
```

如果只想先验证少量样本，可额外添加：

```bash
  --max-source-images 10 \
  --max-guide-images 10 \
  --workers 0 \
  --no-resume
```

## 输出内容

脚本严格按 `cosine_global` 划分为三个互斥子集：

- `cosine_global > 0`：`images/train_cos_more_0` 和 `labels/train_cos_more_0`
- `cosine_global < 0`：`images/train_cos_less_0` 和 `labels/train_cos_less_0`
- `cosine_global == 0`：`images/train_cos_equal_0` 和 `labels/train_cos_equal_0`

每个子集还会在输出根目录生成同名的 `.txt` 图像清单。三个目录名可以分别通过
`--positive-name`、`--negative-name` 和 `--zero-name` 修改。运行配置、分数和
汇总信息保存在 `gradient_scores_ultralytics_val/` 与 `split_summary.json` 中。
