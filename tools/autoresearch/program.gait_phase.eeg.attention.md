# 步态脑电 attention timing scan

## 目标
- 继续固定任务：`支撑 / 摆动` 二分类
- 固定标签：`gait_phase_reference_provisional_v1_0717_0719`
- 先回答哪一个 `window_seconds × global_lag_ms` 最有信息，再回到更多算法家族

## 当前搜索空间
- `window_seconds = 0.1 / 0.5 / 1.0 / 2.0 / 3.0`
- `global_lag_ms = -100 / 0 / 100 / 250 / 500`
- `model_family = feature_gru_attention / feature_tcn_attention`

## 当前真实在跑的算法代码
- 训练入口脚本：
  [train_gait_phase_eeg_classifier.py](/Users/mac/Code/AutoBci/scripts/train_gait_phase_eeg_classifier.py)
- sequence family 的模型分派入口：
  [train_feature_lstm.py](/Users/mac/Code/AutoBci/scripts/train_feature_lstm.py)
- 当前 attention 模型本体：
  [gru_attention_regressor.py](/Users/mac/Code/AutoBci/src/bci_autoresearch/models/gru_attention_regressor.py)
  [tcn_attention_regressor.py](/Users/mac/Code/AutoBci/src/bci_autoresearch/models/tcn_attention_regressor.py)
  [attention_pooling.py](/Users/mac/Code/AutoBci/src/bci_autoresearch/models/attention_pooling.py)

### 代码口径
- 当前不是“整段窗口自由注意力”，而是先把每个相位区间的稳定核心带投到 feature bins 上，再把这个布尔 mask 传给 attention pooling。
- 当前 attention 只在 `feature_gru_attention` 和 `feature_tcn_attention` 两条 family 上启用；其它 family 仍是普通序列模型。
- 当前输出仍是 `2` 类，也就是 `支撑 / 摆动` 二分类，不是连续值回归。

```python
# scripts/train_feature_lstm.py
if family == "feature_gru_attention":
    return GRUAttentionRegressor(...)
if family == "feature_tcn_attention":
    return TCNAttentionRegressor(...)
```

```python
# scripts/train_gait_phase_eeg_classifier.py
def project_safe_band_mask(...):
    core_start = interval_start_idx + 0.25 * interval_length
    core_end = interval_start_idx + 0.75 * interval_length
    # 只把和稳定核心带有重叠的 feature bins 标成 True
```

```python
# scripts/train_gait_phase_eeg_classifier.py
if algorithm_family.endswith("_attention"):
    logits = model(batch_x.to(device), attention_mask=batch_mask.to(device))
else:
    logits = model(batch_x.to(device))
```

```python
# src/bci_autoresearch/models/gru_attention_regressor.py
sequence = x.transpose(1, 2)
hidden, _ = self.gru(sequence)
pooled, _weights = self.pool(hidden, attention_mask=attention_mask)
return self.head(pooled)
```

```python
# src/bci_autoresearch/models/tcn_attention_regressor.py
hidden = self.input_proj(x)
for block in self.blocks:
    hidden = block(hidden)
sequence = hidden.transpose(1, 2)
pooled, _weights = self.pool(sequence, attention_mask=attention_mask)
return self.head(pooled)
```

```python
# src/bci_autoresearch/models/attention_pooling.py
logits = self.score(sequence).squeeze(-1)
logits = logits.masked_fill(~effective_mask, torch.finfo(logits.dtype).min)
weights = torch.softmax(logits, dim=1)
pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
```

## 样本与锚点
- 标签锚点继续来自稳定区中点
- `ambiguous_double_peak` 仍排除，不进入训练样本
- 每个样本额外带一个稳定区核心带：
  - 对当前相位区间取 `25%~75%`
  - 把核心带映射到当前 EEG feature bins
  - attention 只允许在这段核心带内做 masked pooling

## lag 语义
- `lag > 0`：脑电窗口结束在锚点之前
- `lag = 0`：脑电窗口结束在锚点本身
- `lag < 0`：脑电窗口结束在锚点之后，允许少量反馈或对齐误差

## 执行规则
- smoke 全量跑完
- 只晋升 top-2 formal
- 每条 smoke 后都要写 judgment
- smoke 收口后，必须明确写出：
  - 当前最优 `(family, window, lag)` 组合
  - 为什么只晋升前 2 条 formal
  - 若整体仍接近随机，要明确记成 `timing scan 负结果`
