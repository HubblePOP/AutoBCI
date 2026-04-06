# bank QC: walk_matched_v1_64clean_joints

- 通过门禁：yes
- 总 session: 22
- match: 22
- mismatch: 0
- uncertain: 0

## 分 split

- test: total=2 match=2 mismatch=0 uncertain=0
- train: total=18 match=18 mismatch=0 uncertain=0
- val: total=2 match=2 mismatch=0 uncertain=0

## 明细

| session | split | active_bank | candidate_half | score_gap | status | note |
| --- | --- | --- | --- | ---: | --- | --- |
| walk_20240717_01 | train | A | A | 0.284 | match | B has many high-amplitude channels |
| walk_20240717_03 | train | A | A | 0.281 | match | B has many high-amplitude channels |
| walk_20240717_04 | train | A | A | 0.183 | match | B has many high-amplitude channels |
| walk_20240717_05 | train | A | A | 0.330 | match | B has many high-amplitude channels |
| walk_20240717_06 | train | B | B | -0.321 | match | A has many low-variance channels; B has stronger within-bank correlation |
| walk_20240717_07 | train | A | A | 0.334 | match | B has many high-amplitude channels |
| walk_20240717_08 | train | A | A | 0.715 | match | B has many high-amplitude channels; A keeps more channels in the nominal std range |
| walk_20240717_09 | train | A | A | 1.447 | match | B has many high-amplitude channels; A keeps more channels in the nominal std range |
| walk_20240717_10 | train | A | A | 0.669 | match | B has many high-amplitude channels |
| walk_20240717_12 | val | A | A | 0.769 | match | B has many high-amplitude channels |
| walk_20240717_14 | train | A | A | 0.298 | match | B has many high-amplitude channels |
| walk_20240717_16 | test | A | A | 0.678 | match | B has many high-amplitude channels |
| walk_20240719_01 | train | A | A | 1.236 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_02 | train | A | A | 0.953 | match | A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_03 | train | A | A | 1.131 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_04 | train | A | A | 1.003 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_05 | train | A | A | 1.011 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_06 | train | A | A | 1.081 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_07 | val | A | A | 0.780 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_08 | train | A | A | 1.090 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
| walk_20240719_09 | train | A | A | 0.892 | match | B has many low-variance channels; A keeps more channels in the nominal std range |
| walk_20240719_10 | test | A | A | 1.280 | match | B has many low-variance channels; A has stronger within-bank correlation; A keeps more channels in the nominal std range |
