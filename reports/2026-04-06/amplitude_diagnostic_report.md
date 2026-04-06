# Phase C amplitude diagnostic

- accepted best: `stageC_ridge`

## stageC_random_forest

| joint | gain | bias | r | MAE | RMSE | status | Δgain | Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mcp | 0.3260 | 4.9361 | 0.1592 | 10.0699 | 13.5292 | severe compression | 0.0173 | 0.3068 |
| Elb | 0.3562 | -0.1921 | 0.3623 | 11.2808 | 13.7389 | severe compression | -0.0867 | 0.1620 |
| Kne | 0.3614 | -4.2415 | 0.3254 | 12.7724 | 15.9992 | severe compression | -0.1024 | 0.6830 |
| Wri | 0.4098 | -0.3849 | 0.3638 | 12.3179 | 16.7405 | severe compression | -0.0937 | 0.4188 |
| Ank | 0.4368 | -2.4227 | 0.2817 | 8.3907 | 10.9535 | severe compression | -0.0958 | -0.1138 |
| Hip | 0.4397 | -1.6365 | 0.2502 | 5.0120 | 6.2765 | severe compression | -0.0270 | 0.5686 |
| Mtp | 0.5171 | -0.7771 | 0.3661 | 8.7977 | 11.0993 | moderate compression | -0.0704 | -0.1022 |
| Sho | 0.5177 | -0.1727 | 0.5019 | 3.7442 | 4.6043 | moderate compression | -0.0704 | -0.0560 |

## stageC_xgboost_64

| joint | gain | bias | r | MAE | RMSE | status | Δgain | Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mcp | 0.2726 | 4.9448 | 0.1968 | 9.8100 | 13.3234 | severe compression | -0.0361 | 0.3154 |
| Hip | 0.2828 | -1.4489 | 0.3733 | 4.7737 | 5.9056 | severe compression | -0.1839 | 0.7562 |
| Kne | 0.2916 | -4.3747 | 0.3905 | 12.6546 | 15.7159 | severe compression | -0.1722 | 0.5498 |
| Elb | 0.3011 | -0.2460 | 0.4359 | 11.0730 | 13.4149 | severe compression | -0.1418 | 0.1081 |
| Wri | 0.3523 | -0.2900 | 0.4328 | 11.8655 | 16.2443 | severe compression | -0.1513 | 0.5137 |
| Ank | 0.3931 | -2.0654 | 0.3419 | 8.0165 | 10.5473 | severe compression | -0.1395 | 0.2435 |
| Mtp | 0.4338 | -0.6812 | 0.4396 | 8.3642 | 10.5706 | severe compression | -0.1538 | -0.0062 |
| Sho | 0.4671 | -0.1137 | 0.5396 | 3.6499 | 4.4960 | severe compression | -0.1211 | 0.0031 |

## stageC_xgboost_256

| joint | gain | bias | r | MAE | RMSE | status | Δgain | Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mcp | 0.3715 | 4.3797 | 0.2069 | 9.7578 | 13.2278 | severe compression | 0.0628 | -0.2496 |
| Elb | 0.4178 | -0.1068 | 0.4715 | 10.5167 | 13.0218 | severe compression | -0.0251 | 0.2473 |
| Kne | 0.4199 | -3.8371 | 0.4228 | 12.0987 | 15.2640 | severe compression | -0.0438 | 1.0874 |
| Hip | 0.4322 | -1.4455 | 0.4012 | 4.6222 | 5.8118 | severe compression | -0.0345 | 0.7597 |
| Wri | 0.4671 | -0.9230 | 0.4531 | 11.9588 | 16.0261 | severe compression | -0.0364 | -0.1194 |
| Ank | 0.4732 | -1.6358 | 0.3633 | 7.7717 | 10.4390 | severe compression | -0.0594 | 0.6731 |
| Mtp | 0.5401 | -0.8975 | 0.4745 | 8.1680 | 10.4052 | moderate compression | -0.0474 | -0.2226 |
| Sho | 0.5479 | -0.0636 | 0.5576 | 3.5465 | 4.4160 | moderate compression | -0.0402 | 0.0531 |

## stageC_feature_lstm_seed0

| joint | gain | bias | r | MAE | RMSE | status | Δgain | Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mcp | 0.2784 | 5.1240 | 0.2470 | 9.8185 | 13.2302 | severe compression | -0.0303 | 0.4946 |
| Mtp | 0.5039 | 2.0494 | 0.4506 | 8.4478 | 10.7010 | moderate compression | -0.0837 | 2.7244 |
| Ank | 0.5568 | -2.5886 | 0.3377 | 8.2716 | 10.9320 | moderate compression | 0.0242 | -0.2797 |
| Kne | 0.5940 | -2.7067 | 0.4081 | 11.8308 | 15.4279 | moderate compression | 0.1303 | 2.2178 |
| Hip | 0.5983 | -1.9366 | 0.4836 | 4.4827 | 5.7574 | moderate compression | 0.1316 | 0.2685 |
| Elb | 0.6284 | -2.8091 | 0.4145 | 11.0518 | 14.0615 | moderate compression | 0.1855 | -2.4550 |
| Wri | 0.6755 | -3.1390 | 0.3931 | 13.6474 | 17.5441 | moderate compression | 0.1720 | -2.3354 |
| Sho | 0.7461 | 1.1146 | 0.5449 | 3.7360 | 4.7200 | moderate compression | 0.1580 | 1.2313 |

## stageC_feature_lstm_seed1

| joint | gain | bias | r | MAE | RMSE | status | Δgain | Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mcp | 0.3100 | 5.0948 | 0.2904 | 9.6006 | 13.0747 | severe compression | 0.0013 | 0.4655 |
| Ank | 0.6342 | -2.4014 | 0.3662 | 8.1218 | 10.9101 | moderate compression | 0.1016 | -0.0925 |
| Kne | 0.6392 | -3.5729 | 0.4867 | 11.0690 | 14.8913 | moderate compression | 0.1754 | 1.3516 |
| Mtp | 0.6645 | 0.4368 | 0.5157 | 7.9009 | 10.2219 | moderate compression | 0.0770 | 1.1117 |
| Elb | 0.6709 | -0.5118 | 0.4732 | 10.0960 | 13.3169 | moderate compression | 0.2279 | -0.1576 |
| Sho | 0.6711 | 2.5784 | 0.5758 | 4.0392 | 5.0809 | moderate compression | 0.0830 | 2.6951 |
| Hip | 0.6723 | -2.4831 | 0.5164 | 4.6101 | 5.8948 | moderate compression | 0.2056 | -0.2780 |
| Wri | 0.7252 | -0.4908 | 0.4766 | 12.3051 | 16.4030 | moderate compression | 0.2217 | 0.3128 |

## stageC_feature_lstm_seed2

| joint | gain | bias | r | MAE | RMSE | status | Δgain | Δbias |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Mcp | 0.3218 | 2.6030 | 0.0445 | 10.8053 | 13.3013 | severe compression | 0.0131 | -2.0263 |
| Kne | 0.5425 | -3.5908 | 0.4256 | 11.8280 | 15.3023 | moderate compression | 0.0788 | 1.3337 |
| Ank | 0.5482 | -2.3773 | 0.3035 | 8.3630 | 11.0696 | moderate compression | 0.0155 | -0.0683 |
| Mtp | 0.5756 | -0.0430 | 0.4729 | 8.1099 | 10.4179 | moderate compression | -0.0119 | 0.6320 |
| Wri | 0.5791 | -3.0826 | 0.4052 | 12.8678 | 16.9831 | moderate compression | 0.0756 | -2.2790 |
| Hip | 0.5878 | -1.9585 | 0.5075 | 4.4706 | 5.6646 | moderate compression | 0.1211 | 0.2466 |
| Elb | 0.6916 | -1.7803 | 0.3642 | 11.3297 | 14.6584 | moderate compression | 0.2487 | -1.4261 |
| Sho | 0.7186 | 1.9361 | 0.5531 | 3.9691 | 4.9151 | moderate compression | 0.1305 | 2.0529 |
