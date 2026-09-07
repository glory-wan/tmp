# 将梯度对齐并入 Prompt 优化的第二版方案

## 1. 目标与范围

第二版在 `detection_gligen_cos` 中实现，目标是把当前“生成后计算梯度余弦并按正负划分样本”的离散筛选，改成
Prompt 优化阶段内的可微约束，使学到的 Prompt 更倾向于生成对当前检测器训练有正向作用的样本。

本方案只描述后续修改内容。确认方案之前不修改任何 Python 代码，也不启动生成、训练或梯度计算任务。

第二版需要满足以下结果：

1. Prompt 优化时计算生成样本梯度与 Guide 平均梯度的全局余弦值，并将其加入 Prompt loss。
2. `generate` 后不再执行正、负、零样本划分，也不再复制 `cos_more_0`、`cos_less_0`、`cos_equal_0` 数据目录。
3. 每轮只训练一个检测器，训练集为原始训练集加截至当前轮的全部合成样本。
4. 下一轮使用上一轮这个单一检测器的 `best.pt` 继续挖掘失败样本。
5. 原有 `build_coco_cos_subsets_ultralytics.py` 保持不变，仅作为数值对齐测试和离线实验审计工具，不再属于闭环必经阶段。

## 2. 当前实现及需要改变的位置

当前 `detection_gligen_cos/retrain_runner.py` 实际执行：

```text
baseline / 上一轮 cos_more_0 best.pt
  → mine
  → optimize
  → generate
  → align：逐张计算 cos，并复制成 cos>0、cos<0、cos==0 三组
  → retrain cos_more_0
  → retrain all
  → 下一轮使用 cos_more_0 best.pt
```

当前 Prompt 优化目标为：

```text
L_old = λ_semantic · L_semantic
      + λ_class · L_class_semantic
      - λ_detector · L_detector
```

其中负号表示 Prompt 优化在保持语义和图像结构的同时，主动提高当前检测器的检测损失，从而生成 hard example。但高损失
样本的训练梯度不一定与真实验证集梯度同向，因此生成后还需要 `build_coco_cos_subsets_ultralytics.py` 再做一次筛选。

另一个必须先修正的问题是，`detection_gligen_cos/retrain_runner.py` 当前仍调用
`detection_gligen_sdedit.*` 模块。第二版需要把闭环使用的 mine、optimize、generate 和 internal retrain 入口切换到
`detection_gligen_cos.*`，否则在 cos 目录中的修改不会真正被闭环执行。

## 3. 第二版目标流程

```text
baseline / 上一轮 cos_prompt best.pt
  → mine
  → optimize：计算 Guide 平均梯度，并把梯度余弦加入 Prompt loss
  → generate：使用优化后的 Prompt 生成全部样本
  → retrain：原始训练集 + 第 1..当前轮的全部 synthetic/images
  → 下一轮使用 cos_prompt best.pt
```

`align` 阶段和双训练分支全部移除。建议单一训练输出命名为：

```text
round_<round>_cos_prompt
```

这样可以与第一版的 `round_<round>_cos_more_0` 和 `round_<round>_all` 明确区分。

## 4. 梯度对齐的数学定义

设当前轮检测器参数为 `θ`，只选择最后一个检测头的参数 `θ_head`，与现有梯度筛选脚本保持一致。

### 4.1 Guide 平均梯度

对真实验证 Guide 集计算：

```text
g_guide = (1 / N) · Σ_i ∇_(θ_head) L_det(x_i^guide, y_i^guide; θ)
```

`g_guide` 每轮只计算一次，并在本轮 Prompt 优化的所有 step 中保持冻结。每轮输入模型不同，因此不能跨轮无条件复用。

### 4.2 Prompt step 的生成样本梯度

Prompt token 参数记为 `p`，当前 step 产生的可微合成图像为 `x_syn(p)`：

```text
g_syn(p) = ∇_(θ_head) L_align_det(x_syn(p), y_layout; θ)
```

然后计算与旧筛选脚本相同的全局余弦，而不是逐层余弦的平均：

```text
cos(p) = <g_syn(p), g_guide>
         / (||g_syn(p)|| · ||g_guide|| + ε)
```

### 4.3 新的 Prompt loss

推荐第一版采用与旧 `cos > 0` 判定直接对应的 margin hinge：

```text
L_cos = ReLU(cos_margin - cos(p))

L_new = λ_semantic · L_semantic
      + λ_class · L_class_semantic
      - λ_detector · L_detector
      + λ_cos · L_cos
```

建议初始设置：

```text
cos_loss: hinge
cos_margin: 0.0
cos_weight: 1.0
cos_warmup_steps: max_train_steps 的前 10%
```

含义如下：

- `cos < 0` 时明确惩罚，促使 Prompt 离开会产生负向训练梯度的区域。
- `cos >= 0` 时不继续强迫余弦趋近 1，避免 Prompt 为追求单一方向而损失多样性。
- 前 10% step 将 `λ_cos` 从 0 线性增加到配置值，降低 token 初始化阶段的二阶梯度冲击。
- `-λ_detector · L_detector` 继续提供“难样本”压力，`λ_cos · L_cos` 负责约束这些难样本的训练方向是有益的。

后续消融可以增加 `negative_cosine`，即 `L_cos = 1 - cos`，但不作为首个闭环版本的默认策略。

## 5. 必须与现有筛选脚本保持一致的定义

为了让“Prompt 中的 cos”与历史 `build_coco_cos_subsets_ultralytics.py` 的 cos 可比较，以下定义不能各自实现成不同版本：

1. **参数范围**：按固定顺序使用最后检测头的全部 `named_parameters()`。当前三类 YOLO26n 检查点对应 132 个参数张量、
   242346 个参数；具体数量仍应从实际模型读取和校验，不能写死。
2. **检测损失**：使用当前模型的原生 Ultralytics detection loss；第一阶段只支持本实验实际使用的 YOLO detect。
3. **Guide 集**：使用数据集 `val` 的图像与 YOLO 标签，计算全局平均参考梯度。
4. **图像预处理**：复现旧脚本的 letterbox、114 padding、round 规则和 `scaleup` 行为。
5. **框变换**：letterbox 后同步变换归一化框，不能只变换图像。
6. **数值精度**：检测器、Guide 梯度和余弦计算使用 FP32；Guide 梯度作为 detached tensor 缓存。
7. **余弦方式**：先对所有参数张量累计 dot 和 norm，再得到一个全局 cosine。
8. **模型状态**：BatchNorm 统计量不得在 Guide 计算或 Prompt 优化中更新。

当前 Prompt detector 会把 512×512 图像直接插值到 640×640，而旧筛选配置使用 `scaleup: false`，会把 512×512 图像放在
640×640 的 114 padding 画布中。第二版必须新增“图像与 targets 一起进行可微 letterbox”的路径，否则 Prompt 阶段的
cos 与生成后旧脚本计算的 cos 不是同一个量。

## 6. Prompt step 使用哪些标签

当前 `L_detector` 只使用被抽中的一个 failure instance，这是有意保留的 hard-example 目标。

梯度对齐项建议使用独立的 `y_layout`：

1. 取该 source image 中的完整有效标注；
2. failure target 优先；
3. 使用与最终 GLIGEN 生成一致的 `max_objects_per_image` 和小框过滤规则；
4. 转换成最终生成标签相同的类别 ID 和归一化框；
5. 再进行 detector letterbox 对应的框变换。

因此同一个 Prompt step 中存在两个 detector target：

```text
active_target  → 计算原有 L_detector，维持针对失败实例的难度优化
layout_target  → 计算 g_syn 和 cosine，逼近最终生成样本的完整标签语义
```

这比直接用单个 failure target 计算 cos 更接近生成后筛选脚本面对的多目标合成标签。

## 7. Guide 梯度模块设计

建议新增：

```text
detection_gligen_cos/prompt_gradient_alignment.py
```

该模块只负责可复用的梯度对齐基础能力：

- 解析 Guide 图像和 YOLO detection 标签；若数据集标签是 YOLO 分割多边形，则按现有筛选脚本转换为外接框；
- 实现与旧脚本一致的 letterbox 和 target 变换；
- 按顺序选择检测头参数并输出参数 metadata；
- 计算、保存和加载 Guide 平均梯度；
- 计算可微全局 cosine、dot 和两个 gradient norm；
- 构建缓存 metadata，并严格检查缓存是否可复用；
- 对零范数、NaN 和 Inf 立即报错或记录明确的退化状态。

不直接修改或导入 `build_coco_cos_subsets_ultralytics.py` 作为运行时依赖。根目录脚本是面向命令行和文件物化的工具，直接
依赖它会把 argparse、全局 torch 初始化和复制逻辑带进 Prompt 优化。第二版在 cos package 内实现精简模块，并用数值
一致性测试防止两份定义漂移。

Guide 缓存建议放在：

```text
<workspace>/round_<round>/prompts/gradient_alignment/guide_gradients.pt
<workspace>/round_<round>/prompts/gradient_alignment/guide_metadata.json
```

缓存 metadata 至少包含：

- state schema 和 gradient cache schema；
- 检查点绝对路径及 SHA256；
- Ultralytics 版本、task 和 loss kind；
- detector head 参数名称、shape、numel 和顺序；
- dataset YAML、Guide 图像/标签路径及内容指纹；
- image size、scaleup、Guide batch size；
- 类别名称与类别映射。

任一字段变化都重新计算，不允许静默复用旧梯度。

## 8. 检测器 adapter 修改

修改 `detection_gligen_cos/modeling/ultralytics_detector.py`：

1. 保留现有 `differentiable_loss()`，继续服务 `L_detector`。
2. 新增有序的 detection-head 参数选择接口；其余 detector 参数始终冻结。
3. 新增与旧梯度筛选定义一致的 alignment loss 路径。
4. alignment 路径支持可微 letterbox，并同时返回变换后的 targets。
5. 计算 `g_syn` 时使用：

   ```python
   torch.autograd.grad(
       alignment_detector_loss,
       selected_head_parameters,
       create_graph=True,
       retain_graph=True,
       allow_unused=True,
   )
   ```

6. 未使用参数的梯度按零张量处理，顺序必须与 Guide cache 一致。
7. detector head 不加入 Prompt optimizer；每 step 后清理可能产生的 `.grad`，并断言 detector 权重没有变化。

`create_graph=True` 是关键：如果对 `g_syn` 做 detach，日志里虽然能看到 cos，但 cos 无法把梯度传回生成图像和 Prompt token。

## 9. Prompt 优化器修改

修改 `detection_gligen_cos/optimize_object_prompts_ultralytics.py`：

1. 增加 Guide 输入、cos loss、margin、warmup、target scope、cache 等 CLI 参数。
2. 加载 detector 后立即选择 alignment 参数，并计算或恢复本轮 Guide 梯度。
3. 构造完整 `layout_target`。
4. 每个 step 在原有 semantic、class semantic、detector loss 后计算 `g_syn`、cosine 和 `L_cos`。
5. 按第 4 节公式合成 loss，再只更新当前 group 的 placeholder token。
6. 日志新增：

   ```text
   cos
   cos_loss
   cos_weight_effective
   generated_gradient_norm
   guide_gradient_norm
   dot
   active_target_count
   layout_target_count
   ```

7. 对每个 prompt group 维护 running count、mean、min、max、positive rate，并在保存 checkpoint 时写入 metadata。
8. `object_prompts.json` 新增 alignment 配置、Guide cache metadata/hash、参数范围、累计 cos 统计和实现模式。
9. Prompt checkpoint 恢复时继续复用 token embeddings；Guide cache 只在 metadata 完全匹配时复用。

Prompt optimizer 的主进程模块必须改为：

```text
detection_gligen_cos.optimize_object_prompts_ultralytics
```

## 10. 闭环 runner 修改

修改 `detection_gligen_cos/retrain_runner.py`：

### 10.1 模块命名空间隔离

闭环子进程统一调用：

```text
detection_gligen_cos.mine_failures_ultralytics
detection_gligen_cos.optimize_object_prompts_ultralytics
detection_gligen_cos.generate_gligen_sdedit_examples
detection_gligen_cos.retrain_runner
```

第一阶段只修改闭环直接使用的入口，不批量改动不参与闭环的旧实验脚本。

### 10.2 删除生成后 alignment 阶段

移除 runner 中以下闭环行为：

- `run_alignment()` 调用；
- positive/negative/zero subset name 生成；
- gradient score 目录初始化和恢复；
- 数据集根目录中的三组目录冲突检查；
- `split_summary.json` 对闭环状态的依赖；
- `build_coco_cos_subsets_ultralytics.py` 必须存在的启动校验。

### 10.3 单分支 retrain

每轮生成一个训练 YAML：

```yaml
path: <original-dataset-root>
train:
  - images/train2017
  - <workspace>/round_1/synthetic/images
  - <workspace>/round_2/synthetic/images
  # ...直到当前 round
val: images/val2017
names: ...
```

合成目录使用绝对路径，Ultralytics 根据路径中的 `images` 自动找到同级 `labels`，不再把合成数据复制到原始数据集根目录。

每轮仍从 `model_yaml` 新建检测器，并在“真实数据 + 截至当前轮全部合成数据”上从头训练，保持现有两分支实验的初始化原则，
只是将分支数量从 2 减为 1。

### 10.4 下一轮模型和最终模型

```text
model_for_round(round=1) = baseline
model_for_round(round>1) = previous_round.retrain.best
final_model = last_round.retrain.best
```

内部 retrain CLI 不再需要 `--retrain-group`。

## 11. 配置调整建议

删除旧的顶层 `gradient_alignment.script`、subset name、score dir 和 copy worker 配置，把仍需要的 Guide 设置移到
`prompt_optimization.gradient_alignment`：

```yaml
prompt_optimization:
  # 保留现有 Prompt 参数
  semantic_weight: 1.0
  class_semantic_weight: 0.0
  detector_weight: 0.1

  gradient_alignment:
    enabled: true
    guide_root: null          # null 表示使用 dataset YAML 解析出的数据集根目录
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

retrain:
  run_name_template: round_{round}_cos_prompt
```

配置归一化阶段需要验证：

- `weight >= 0`；
- `margin` 在 `[-1, 1]`；
- `warmup_steps >= 0`；
- `guide_batch_size`、`image_size` 和 `workers` 合法；
- 当前版本只接受 `task=detect`、`parameter_scope=detection_head`、`implementation=exact`；
- `enabled=true` 时 Guide 图像和标签必须存在且非空。

## 12. 状态 JSON 与恢复策略

闭环结构发生不兼容变化，建议把 `STATE_SCHEMA_VERSION` 从 1 升到 2，并明确拒绝直接恢复第一版 state。第一版实验继续使用原
runner/state，新实验必须使用新 workspace，避免把两种语义混在一起。

每轮 state 简化为：

```text
round
├── mine
├── optimize
│   ├── latest prompt checkpoint
│   ├── completed_steps
│   ├── guide cache path/hash
│   └── alignment metadata/statistics
├── generate
└── retrain
    ├── epoch
    ├── data YAML
    ├── run_dir
    ├── last.pt
    └── best.pt
```

恢复规则：

1. `optimize` 中断：从最新 token checkpoint 恢复；Guide metadata 一致时复用 cache。
2. `generate` 中断：沿用现有逐图 manifest 原子恢复。
3. `retrain` 中断：从单一 `last.pt` 恢复，并持续写回 epoch。
4. 上一轮模型、Guide cache 或关键 alignment 配置变化时，拒绝复用对应阶段结果。

## 13. 测试与验证计划

### 13.1 单元测试

新增或修改以下测试：

1. **namespace 测试**：runner 构造的所有闭环命令均指向 `detection_gligen_cos.*`。
2. **letterbox 一致性**：新模块的图像与框结果和旧 `YoloBoxDataset` 一致。
3. **参数一致性**：新 adapter 与旧脚本选择的参数名称、shape、顺序完全一致。
4. **cos 数值一致性**：同一模型、同一 Guide cache、同一图像和标签下，新模块 detached cosine 与旧脚本结果在容差内一致。
5. **可微性测试**：只启用 `L_cos` 时，active placeholder token 获得有限且非零的梯度。
6. **冻结测试**：优化 step 前后 detector 参数完全相同，非 active token 和原始词表保持不变。
7. **缓存测试**：相同 metadata 命中缓存；模型、标签、参数顺序或预处理变化会使缓存失效。
8. **退化测试**：Guide 零范数、source 非有限梯度、空 Guide、类别不匹配时给出明确错误。
9. **runner 状态测试**：state 中不存在 `align` 和 `retrain.groups`，只执行一次 retrain。
10. **累积数据集测试**：第 N 轮 YAML 恰好包含原始训练集和第 1..N 轮 synthetic 目录，每个目录只出现一次。
11. **恢复测试**：Prompt、生成和单分支训练分别中断后可恢复；schema 1 state 被明确拒绝。

### 13.2 GPU 冒烟测试

在完整闭环前先执行最小规模测试：

```text
Guide images: 2～4
failure samples: 1～2
Prompt steps: 2
generation samples: 2
retrain epochs: 1
workers: 0
```

必须确认：

- exact cosine 可以二阶反传；
- token gradient 非零且有限；
- detector 权重没有变化；
- 无 NaN、Inf 和不可接受的显存峰值；
- Guide cache 能在恢复时复用；
- 闭环跳过 align，并只生成一个 retrain run。

### 13.3 消融实验

为了区分“去掉生成后筛选”和“cos Prompt 优化”两个变量，至少比较：

| 实验 | Prompt cos loss | 生成后筛选 | retrain 数据 |
| --- | --- | --- | --- |
| V1 基线 | 关闭 | 开启 | cos>0 与 all 两分支 |
| V2-control | 关闭 | 关闭 | 全部生成样本，单分支 |
| V2-cos | 开启 | 关闭 | 全部生成样本，单分支 |

建议记录：

- 验证集 P、R、mAP50、mAP50-95；
- Prompt step 的 cosine mean/min/positive rate；
- 生成池的 detector loss、类别和 bbox 分布；
- 每轮运行时间与峰值显存；
- 使用旧脚本对生成池做**离线只评分、不筛选、不参与闭环**的 cos 分布审计。

离线审计用于验证 Prompt 中的优化信号能否迁移到最终 GLIGEN-SDEdit 生成结果，不恢复旧的生成后分类流程。

## 14. 主要风险和回退路径

### 14.1 二阶梯度显存和算子支持

精确 cosine 需要 `create_graph=True`，会对 detector head gradient 再反传到生成图像和 Prompt token，显存与时间开销明显高于
当前 Prompt 优化。先通过两 step 冒烟测试，不能直接启动完整 1000-step/多轮实验。

依次采用以下降本方式：

1. 只保留当前已经限定的 detection head 参数范围；
2. 降低 Prompt 优化的 denoise step 或 resolution 做功能验证；
3. 必要时增加 `alignment_every_n_steps`，但默认仍为每 step；
4. 若确实出现不支持二阶梯度的算子，再提出有限差分 directional-gradient surrogate 方案。

有限差分只能保证优化 dot 的方向与 cosine 符号一致，不能视为精确 cosine。未经再次确认，不自动从 exact 模式静默降级。

### 14.2 Prompt 优化生成方式与最终生成方式不完全相同

当前 Prompt 优化使用可微 ROI inpainting，最终生成使用 GLIGEN full-image SDEdit。即使优化阶段 cosine 提升，也不能保证最终
GLIGEN 图像的 cosine 同幅度提升。因此必须保留 V2-cos 与 V2-control 的生成池离线审计，但审计结果不能参与样本选择。

### 14.3 hard loss 与 alignment loss 竞争

`-L_detector` 追求高损失，`L_cos` 追求有益方向，两者可能竞争。使用 cos warmup，并先做
`cos_weight ∈ {0.1, 0.5, 1.0}` 的短程实验，再固定完整闭环参数。

### 14.4 全局 Guide 梯度的类别偏置

第一版严格复现现有全局 Guide 平均梯度，不同时引入 class-conditioned Guide，避免一次修改多个变量。如果后续发现少数类别被
全局梯度淹没，再单独设计按 class/group 的 Guide 梯度作为第三版实验。

## 15. 建议实施顺序

确认本方案后，按以下顺序修改：

1. 切换 cos runner 的闭环模块命名空间，并更新对应 mock 测试。
2. 实现 `prompt_gradient_alignment.py` 和 letterbox/参数/cos 数值一致性测试。
3. 扩展 detector adapter，先完成 detached cosine 与旧脚本对齐。
4. 接入 `create_graph=True`，完成 token 二阶梯度和 detector 冻结测试。
5. 将 `L_cos`、日志、metadata 和 Guide cache 接入 Prompt optimizer。
6. 简化 runner 为 `mine → optimize → generate → retrain`，改为单一累积 synthetic 数据集。
7. 升级 state schema、恢复逻辑和配置文件。
8. 更新 `retrain_runner.md`，运行单元测试和最小 GPU 冒烟测试。
9. 冒烟结果通过后，再运行 V2-control/V2-cos 短程消融；确认权重后启动完整闭环。

## 16. 本次迭代明确不做的内容

- 不修改根目录 `build_coco_cos_subsets_ultralytics.py`。
- 不在生成后按 cos 正负分类、复制或删除图像。
- 不自动迁移第一版 state 或复用第一版 workspace。
- 不同时加入 class-conditioned Guide、动态阈值、样本权重训练等额外变量。
- 不修改 Ultralytics 核心源码；所有逻辑放在 `detection_gligen_cos` adapter、Prompt optimizer 和 runner 中。
- 未通过 exact 二阶梯度冒烟前，不启动完整多轮训练。
