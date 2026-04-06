# question A: mean 伪迹排查

## 结果

| run | val r | test r | test MAE | test RMSE |
| --- | ---: | ---: | ---: | ---: |
| stageA_ridge_absmean_rms | 0.1574 | 0.0988 | 9.6504 | 12.0839 |
| stageA_ridge_mean_only | 0.2219 | 0.2296 | 9.3815 | 11.8333 |
| stageA_ridge_mean_absmean_rms | 0.2254 | 0.1828 | 9.4578 | 11.9766 |
| stageA_ridge_session_center | 0.2254 | 0.1828 | 9.4583 | 11.9767 |
| stageA_ridge_target_shuffle | 0.0019 | 0.0126 | 9.7868 | 12.3458 |
| stageA_ridge_target_shift | 0.1414 | 0.0971 | 9.6664 | 12.1711 |

## 判断

- `mean only` 已经接近当前最好结果，`mean` 是这条线里最强的单项特征。
- `session_center` 前后 `val r` 只差 -0.000032。
- `target_shuffle` 后 `val r / best` 只剩 0.008。
- `target_shift(10s)` 后 `val r / best` 还有 0.627。
- 结论：当前 `mean` 更像是任务相关信息和慢变化成分混在一起，先不要把它讲成纯神经特征。
