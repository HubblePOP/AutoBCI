# promotion decision

- 当前决定：`hold_mainline_ridge`
- 主线 accepted best 继续保持：`stageC_ridge`
- `feature-LSTM` gate：`pass`，中位数 `val r = 0.4227`，`test r = 0.3483`
- 更强的单次候选：`stageC_xgboost_256`，`val r = 0.4329`，`test r = 0.3748`
- 处理规则：XGBoost 还没有复验，所以这轮先不提升任何候选。
- 上限线 `feature-LSTM`：`test r = 0.4745`
- 下一步：先补 `stageC_xgboost_256` 复验，再重新决定是否升主线。