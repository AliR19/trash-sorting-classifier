# Results of the clean top-to-bottom Colab run (T4 GPU, seed 42)

Copied from the output of `notebooks/TME6015_FinalProject.ipynb`. Sets: `test` = 375 TrashNet photos (trash n=20), `external` = 480 held-out Garbage Dataset photos (80 per class; called the GD hold-out in the README and report: other GD photos are in training, so it is not an independent benchmark), `bgshift` = 162 test photos on synthetic new backgrounds (same synthetic pipeline as the synthetic training images, so it measures tolerance to that recipe, not real-world robustness).

## Macro-F1

| model | test | external | bgshift |
|---|---|---|---|
| B0_plain_real | 0.909 | 0.915 | 0.434 |
| B1_plain_real+synthetic | 0.907 | 0.919 | 0.612 |
| D1_plain+dropout_real+syn | 0.919 | 0.917 | 0.595 |
| B2_plain_mobilenetv3 | 0.884 | 0.913 | 0.626 |
| E0_reg_real_only | 0.888 | 0.917 | 0.390 |
| E1_reg_real+synthetic | 0.859 | 0.927 | 0.618 |
| E2_reg_mobilenetv3 | 0.887 | 0.917 | 0.671 |
| ENS_plain(B1+B2) | 0.898 | 0.927 | 0.647 |
| ENS_plain(B0+B1+B2) | 0.910 | 0.923 | 0.639 |
| ENS_reg(E0+E1+E2) | 0.908 | 0.931 | 0.636 |
| **ENS_reg(E1+E2) (final)** | 0.883 | 0.933 | 0.673 |

## Accuracy and worst-class recall

| model | test acc | test worst-class recall | external acc | external worst | bgshift acc | bgshift worst |
|---|---|---|---|---|---|---|
| B0_plain_real | 0.933 | 0.60 | 0.915 | 0.86 | 0.488 | 0.17 |
| B1_plain_real+synthetic | 0.928 | 0.65 | 0.919 | 0.89 | 0.630 | 0.54 |
| D1_plain+dropout_real+syn | 0.931 | 0.75 | 0.917 | 0.89 | 0.623 | 0.46 |
| B2_plain_mobilenetv3 | 0.907 | 0.60 | 0.912 | 0.86 | 0.660 | 0.50 |
| E0_reg_real_only | 0.912 | 0.65 | 0.917 | 0.89 | 0.432 | 0.27 |
| E1_reg_real+synthetic | 0.896 | 0.50 | 0.927 | 0.90 | 0.648 | 0.38 |
| E2_reg_mobilenetv3 | 0.904 | 0.65 | 0.917 | 0.89 | 0.704 | 0.46 |
| ENS_plain(B1+B2) | 0.923 | 0.60 | 0.927 | 0.90 | 0.679 | 0.54 |
| ENS_plain(B0+B1+B2) | 0.928 | 0.65 | 0.923 | 0.89 | 0.673 | 0.54 |
| ENS_reg(E0+E1+E2) | 0.925 | 0.65 | 0.931 | 0.89 | 0.685 | 0.38 |
| **ENS_reg(E1+E2) (final)** | 0.912 | 0.55 | 0.933 | 0.90 | 0.710 | 0.46 |

## Training behaviour (last epoch)

| model | train acc | val acc | gap | train loss | val loss |
|---|---|---|---|---|---|
| B0_plain_real | 1.000 | 0.922 | 0.078 | 0.006 | 0.305 |
| B1_plain_real+synthetic | 1.000 | 0.909 | 0.091 | 0.008 | 0.356 |
| B2_plain_mobilenetv3 | 0.999 | 0.904 | 0.095 | 0.005 | 0.448 |
| E0_reg_real_only | 0.981 | 0.912 | 0.069 | 0.523 | 0.346 |
| E1_reg_real+synthetic | 0.977 | 0.904 | 0.073 | 0.532 | 0.367 |
| E2_reg_mobilenetv3 | 0.991 | 0.904 | 0.087 | 0.496 | 0.358 |

(Train loss of the regularized models includes label smoothing, so it is not comparable with the plain ones. The per-epoch history of every model is in `history.json`.)

## Final-model choice (validation sets only, no test data)

Average of macro-F1 on the clean validation set (615 photos) and on the new-background validation set (168 photos):

| model | clean | new background | average |
|---|---|---|---|
| ENS_reg(E1+E2) | 0.909 | 0.657 | **0.783** |
| B2_plain_mobilenetv3 | 0.905 | 0.642 | 0.773 |
| ENS_plain(B0+B1+B2) | 0.914 | 0.612 | 0.763 |
| ENS_reg(E0+E1+E2) | 0.912 | 0.612 | 0.762 |
| ENS_plain(B1+B2) | 0.903 | 0.609 | 0.756 |
| E1_reg_real+synthetic | 0.897 | 0.613 | 0.755 |
| E2_reg_mobilenetv3 | 0.898 | 0.609 | 0.753 |
| B1_plain_real+synthetic | 0.905 | 0.595 | 0.750 |
| D1_plain+dropout_real+syn | 0.905 | 0.579 | 0.742 |
| B0_plain_real | 0.915 | 0.460 | 0.688 |
| E0_reg_real_only | 0.908 | 0.461 | 0.685 |

## Targets and published baselines

Published (accuracy): TrashNet, pretrained GoogLeNet, random 10% hold-out: 0.912, and 0.954 with augmentation (Yu & Grammenos, arXiv 2108.06274); Garbage Dataset, 10 classes: 0.93 to 0.96 for ResNet/EfficientNetV2 (arXiv 2602.10500).

Targets used: accuracy >= 0.91 and worst-class recall >= 0.75. The final model meets the accuracy target on test (0.912) and on external (0.933), meets the worst-class target on external (0.90), and does **not** meet it on test (trash recall 0.55, n=20, 95% interval 0.34 to 0.74). Only D1 reaches 0.75 on test (exactly at the threshold, with n=20).

## Final model with 95% Wilson intervals

| set | n | accuracy | 95% interval |
|---|---|---|---|
| test | 375 | 0.912 | 0.879 to 0.937 |
| external | 480 | 0.933 | 0.907 to 0.952 |
| bgshift | 162 | 0.710 | 0.636 to 0.774 |

Per-class recall with intervals (test / external / bgshift):

| class | test | external | bgshift |
|---|---|---|---|
| cardboard | 0.90 (0.80-0.95), n=60 | 0.96 (0.90-0.99), n=80 | 0.46 (0.23-0.71), n=13 |
| glass | 0.88 (0.79-0.94), n=75 | 0.95 (0.88-0.98), n=80 | 0.68 (0.52-0.80), n=40 |
| metal | 1.00 (0.94-1.00), n=61 | 0.94 (0.86-0.97), n=80 | 0.88 (0.73-0.95), n=34 |
| paper | 0.98 (0.92-0.99), n=87 | 0.90 (0.81-0.95), n=80 | 0.65 (0.46-0.81), n=26 |
| plastic | 0.90 (0.81-0.95), n=72 | 0.91 (0.83-0.96), n=80 | 0.73 (0.57-0.85), n=37 |
| trash | 0.55 (0.34-0.74), n=20 | 0.94 (0.86-0.97), n=80 | 0.67 (0.39-0.86), n=12 |

## Segmentation experiment (isolate with U2-Net, then classify; 40 photos per class or fewer)

| set | input | n | accuracy | macro-F1 |
|---|---|---|---|---|
| test | whole photo | 220 | 0.891 | 0.871 |
| test | segment then classify | 220 | 0.732 | 0.711 |
| external | whole photo | 240 | 0.925 | 0.925 |
| external | segment then classify | 240 | 0.838 | 0.836 |
| bgshift | whole photo | 162 | 0.710 | 0.673 |
| bgshift | segment then classify | 162 | 0.611 | 0.574 |

Segmentation hurt on all three sets, so the final pipeline classifies the whole photo.
