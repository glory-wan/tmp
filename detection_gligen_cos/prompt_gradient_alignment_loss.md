# 梯度对齐如何融入 Prompt 优化损失

当前实现把“生成样本梯度与 Guide 样本梯度的余弦相似度”直接作为 Prompt 优化损失的一部分，不再等样本生成后再按 `cos > 0` 对图像进行分组。

## 1. 总损失

每一步 Prompt 优化使用的损失为：

```text
L_prompt = λ_sem × L_sem
         + λ_cls × L_cls
         - λ_det × L_hard
         + λ_cos_eff × L_cos
```

- `L_sem`：生成区域与原目标区域之间的语义/结构保持损失。
- `L_cls`：可选的 CLIP 类别语义损失。
- `L_hard`：检测器在当前失败目标上的检测损失。前面的负号表示优化 Prompt 时会增大该损失，以生成困难样本。
- `L_cos`：梯度对齐损失，用于约束困难样本的训练梯度与 Guide 数据的训练方向一致。
- `λ_cos_eff`：经过 warmup 后的梯度对齐有效权重。

## 2. 梯度余弦相似度

每轮开始时，先在 Guide 数据上计算并缓存检测头的平均梯度：

```text
g_guide = mean(∇θ L_det(x_guide, y_guide))
```

每一步生成可微图像后，再用完整布局标注计算该生成样本对检测头参数的梯度：

```text
g_syn(p) = ∇θ L_det(x_syn(p), y_layout)
```

其中 `p` 是正在优化的 Prompt token，`θ` 是检测器最后一个检测头的参数。随后计算全局余弦相似度：

```text
cos = <g_syn, g_guide> / (||g_syn|| × ||g_guide|| + ε)
```

代码使用 `create_graph=True` 计算 `g_syn`，因此 `L_cos` 能够继续反向传播到生成图像和 Prompt token。这是一个二阶梯度过程。

## 3. 余弦损失与 warmup

当前使用 hinge 形式：

```text
L_cos = max(0, alignment_margin - cos)
```

默认 `alignment_margin = 0` 时：

- `cos < 0`：产生惩罚，推动 Prompt 生成梯度方向与 Guide 梯度方向一致的样本。
- `cos >= 0`：该项为 0，不再施加额外惩罚。

梯度对齐权重在前 `alignment_warmup_steps` 步内从 0 线性增加到 `alignment_weight`，避免优化初期该项突然主导总损失。

## 4. 实际优化对象

检测器只用于计算检测损失和梯度方向，其参数不会被优化；反向传播后检测器梯度会被清除。最终由优化器更新的仍然只有当前激活的 Prompt 占位 token。

整体流程为：

```text
Guide 数据计算平均梯度并缓存
          ↓
Prompt 生成可微图像
          ↓
计算困难样本损失、语义损失和梯度余弦损失
          ↓
组合为 L_prompt 并反向传播
          ↓
仅更新 Prompt token
```

主要实现位于：

- `optimize_object_prompts_ultralytics.py`：组合 Prompt 总损失。
- `prompt_gradient_alignment.py`：计算 Guide 梯度、生成样本梯度及全局余弦相似度。
- `modeling/ultralytics_detector.py`：提供可微的 Ultralytics 检测损失。
