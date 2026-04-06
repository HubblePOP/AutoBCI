# Phase C segment diagnostic

- 固定主片段：`walk_20240717_16 @ 217.6s-229.6s`
- 自动难片段：`walk_20240717_16 @ 158.4s-170.4s`

## fixed_main_segment

- session: `walk_20240717_16`
- time: `217.6s-229.6s`

| model | joint | local r | gain | bias | true amp | pred amp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stageC_ridge | Kne | 0.3285 | 0.2693 | -8.3081 | 54.9926 | 24.5999 |
| stageC_ridge | Wri | 0.2539 | 0.4068 | 2.7621 | 54.5917 | 20.3484 |
| stageC_ridge | Mcp | 0.0682 | 0.2367 | 3.7669 | 50.1781 | 10.2309 |
| stageC_xgboost_256_seed2 | Kne | 0.4279 | 0.2666 | -6.5579 | 54.9926 | 26.6832 |
| stageC_xgboost_256_seed2 | Wri | 0.5262 | 0.4207 | 0.4732 | 54.5917 | 26.8601 |
| stageC_xgboost_256_seed2 | Mcp | 0.2747 | 0.2369 | 4.4967 | 50.1781 | 15.1287 |
| stageC_feature_lstm_seed0 | Kne | 0.5390 | 0.4026 | -7.8293 | 54.9926 | 28.8016 |
| stageC_feature_lstm_seed0 | Wri | 0.3676 | 0.5694 | 3.8114 | 54.5917 | 34.7248 |
| stageC_feature_lstm_seed0 | Mcp | 0.1797 | 0.1922 | 3.8494 | 50.1781 | 10.9994 |

## auto_hard_segment

- session: `walk_20240717_16`
- time: `158.4s-170.4s`
- ridge reference: `mean_local_r=-0.2403`, `mean_true_amplitude=46.4078`

| model | joint | local r | gain | bias | true amp | pred amp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| stageC_ridge | Kne | -0.3253 | 0.2556 | -8.0478 | 63.2313 | 23.4752 |
| stageC_ridge | Wri | -0.0957 | 0.3845 | 2.3161 | 61.0176 | 29.0174 |
| stageC_ridge | Mcp | -0.0798 | 0.2153 | 3.0652 | 53.6975 | 14.2515 |
| stageC_xgboost_256_seed2 | Kne | 0.3112 | 0.2009 | -8.4552 | 63.2313 | 16.2833 |
| stageC_xgboost_256_seed2 | Wri | 0.0466 | 0.3415 | 2.4437 | 61.0176 | 18.4666 |
| stageC_xgboost_256_seed2 | Mcp | 0.0313 | 0.2004 | 5.7353 | 53.6975 | 11.6908 |
| stageC_feature_lstm_seed0 | Kne | -0.0161 | 0.3861 | -6.8090 | 63.2313 | 29.4010 |
| stageC_feature_lstm_seed0 | Wri | 0.0669 | 0.6147 | 2.3502 | 61.0176 | 34.8741 |
| stageC_feature_lstm_seed0 | Mcp | 0.0739 | 0.2266 | 4.3236 | 53.6975 | 12.7440 |
