# Initial experiment

Run: 2026-10-25. Python 3.12, CPU, seed 42. Maximum 30 epochs, patience five.

The neural scorer was selected by validation rating RMSE. Test results were reported after selection; no test-driven retuning was performed.

| Model | Selected epoch | Validation RMSE | Test RMSE | Test MAE | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| popularity | — | — | 0.999351 | 0.779847 | 0.041694 | 0.040152 |
| mf | 4 | 0.8642103850440855 | 0.882423 | 0.677161 | 0.056783 | 0.052617 |
| neural | 2 | 0.8621169193657247 | 0.882299 | 0.674132 | 0.024681 | 0.024844 |

MF ranked relevant held-out movies better than the neural scorer. The neural scorer had slightly lower rating error. These are different objectives; selection by one does not guarantee success on the other.

Ranking covers 557 existing users with positive supported test items, among 8,255 training-catalog movies. Test ratings >=4 are relevant; prior training/validation movies are excluded. Unknown holdout movies are excluded and counted in the README.

This does not evaluate unseen-user adaptation, group satisfaction, or the discovery rule. Splits are chronological per user, not a single global time split.

Selected checkpoint SHA-256: `0532bf0c8effbc9d047e6567cc69400d7de6f52789e44d765c50653bb165ebd6`.

Downloaded archive SHA-256: `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.

Source: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

The upstream README and usage conditions remain in `data/raw/README.txt`. The archive is a development snapshot, not a permanent benchmark.

## mf training history

| Epoch | Training MSE | Validation RMSE |
|---|---:|---:|
| 1 | 0.970439 | 0.956635 |
| 2 | 0.761472 | 0.886756 |
| 3 | 0.610349 | 0.866516 |
| 4 | 0.489733 | 0.864210 |
| 5 | 0.397408 | 0.866124 |
| 6 | 0.336584 | 0.871164 |
| 7 | 0.296080 | 0.876891 |
| 8 | 0.267656 | 0.881742 |
| 9 | 0.247825 | 0.885468 |

## neural training history

| Epoch | Training MSE | Validation RMSE |
|---|---:|---:|
| 1 | 0.841461 | 0.873454 |
| 2 | 0.680296 | 0.862117 |
| 3 | 0.612763 | 0.866549 |
| 4 | 0.485970 | 0.890491 |
| 5 | 0.336088 | 0.924383 |
| 6 | 0.256178 | 0.939249 |
| 7 | 0.213137 | 0.949990 |
