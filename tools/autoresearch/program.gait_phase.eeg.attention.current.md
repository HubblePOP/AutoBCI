# 当前执行合同：步态脑电 attention timing scan

## 本轮问题
不是继续扩算法家族，而是先回答：
- 哪个窗长最有信息
- signed lag 里是不是存在更好的 timing
- attention 版 GRU/TCN 能不能比 plain baseline 更稳

## runner 边界
- 只允许：
  - `feature_gru_attention`
  - `feature_tcn_attention`
- 不回头改步态标签
- 不混入更多 family
- 不把动态锚点做成模型外重新选锚点

## 当前真实在跑的算法代码
- 训练入口：
  [train_gait_phase_eeg_classifier.py](/Users/mac/Code/AutoBci/scripts/train_gait_phase_eeg_classifier.py)
- family 分派：
  [train_feature_lstm.py](/Users/mac/Code/AutoBci/scripts/train_feature_lstm.py)
- attention 模型本体：
  [gru_attention_regressor.py](/Users/mac/Code/AutoBci/src/bci_autoresearch/models/gru_attention_regressor.py)
  [tcn_attention_regressor.py](/Users/mac/Code/AutoBci/src/bci_autoresearch/models/tcn_attention_regressor.py)
  [attention_pooling.py](/Users/mac/Code/AutoBci/src/bci_autoresearch/models/attention_pooling.py)

### 代码结构
- `feature_gru_attention`：先过 GRU，再用 masked attention pooling 收成一个向量，最后接 `2` 类输出头。
- `feature_tcn_attention`：先过因果 TCN block，再用 masked attention pooling 收成一个向量，最后接 `2` 类输出头。
- attention mask 不是随便学出来的，而是由 `project_safe_band_mask(...)` 把相位区间中间 `25%~75%` 的稳定核心带映射到当前 feature bins。
- 只有 `*_attention` family 才会把 `attention_mask` 喂给模型；普通 family 还是按普通序列模型跑。

```python
if family == "feature_gru_attention":
    return GRUAttentionRegressor(...)
if family == "feature_tcn_attention":
    return TCNAttentionRegressor(...)
```

```python
core_start = interval_start_idx + 0.25 * interval_length
core_end = interval_start_idx + 0.75 * interval_length
```

```python
if algorithm_family.endswith("_attention"):
    logits = model(batch_x.to(device), attention_mask=batch_mask.to(device))
```

## 物化口径
- 运行清单来自 `tracks.gait_phase_eeg_attention.json`
- smoke 总数固定为 `50`
- formal 总数固定为 `2`

## 搜索链
- 起步先搜 1 轮
- 搜索主题只围绕：
  - gait EEG timing
  - premovement timing
  - fixed-lag decoding
  - temporal attention for gait decoding
- 每条 smoke 完成后更新 judgment

## 负结果口径
如果 top-2 formal 之后仍整体接近随机：
- 报告必须写成 `timing scan 负结果`
- 不能再写成“只做完了算法摸底”
