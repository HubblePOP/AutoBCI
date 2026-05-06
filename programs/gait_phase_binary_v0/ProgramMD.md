# ProgramMD: gait_phase_binary_v0

- version: 0.1
- status: frozen
- task_type: binary_classification
- primary_metric: test_balanced_accuracy

## 研究目标
用脑电判断步态 support / swing 二分类是否稳定可解

## 数据划分
- unit: session
- train sessions: 18
- val sessions: walk_20240717_12, walk_20240719_07
- test sessions: walk_20240717_16, walk_20240719_10

## 标签定义与风险
- source: gait_phase_reference_provisional_v1_0717_0719
- known risks: short_intervals, pending_manual_review, historical_safe_band_filtering_dependency
- acceptance note: 本任务先验证当前 operational label，不声明真实生理标签已可靠。
- 注意：当前 v1 标签存在大量短 interval 风险，历史高分依赖 safe-band filtering。

## 搜索空间
- windows_seconds: [0.5, 1.0, 2.0, 3.0]
- lags_ms: [0, 100, 250, 500]
- model_families: baseline_logistic, feature_tcn, feature_gru

## 禁区
- change_task_type
- change_primary_metric
- change_split_without_amendment
- modify_raw_data
- overwrite_existing_result
- read_director_scratchpad_from_judge

## 不确定性
- v1 标签包含大量极短 swing interval，需要 Judge 在复评中标注风险。
- 历史 0.7375 高分依赖 historical safe-band filtering，不能直接当作宽口径稳定最好结果。
