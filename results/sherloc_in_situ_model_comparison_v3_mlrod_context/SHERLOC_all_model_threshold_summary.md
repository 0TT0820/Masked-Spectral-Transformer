# SHERLOC In-Situ All-Model Validation Summary

Run directory: `publication_repo/results/sherloc_in_situ_model_comparison_v3_mlrod_context`

This experiment evaluates the reviewer-requested model families on pooled labeled SHERLOC in-situ spectra. It is a within-domain SHERLOC random-split validation, not an independent target-region transfer test. The 730 labeled SHERLOC spectra were repeatedly split using seeds 2024, 2025, and 2026. For each split, 467 spectra were used for training, 117 for internal tuning, and 146 for held-out validation.

## Aggregate Metrics Across Three Random Splits

| Model | Accuracy mean | Accuracy SD | All-label macro-F1 mean | Present-label macro-F1 mean | Weighted-F1 mean | Internal-tune macro-F1 mean |
|---|---:|---:|---:|---:|---:|---:|
| MST | 0.767 | 0.090 | 0.429 | 0.600 | 0.782 | 0.592 |
| PLS-DA | 0.765 | 0.032 | 0.323 | 0.452 | 0.746 | 0.456 |
| PCA-SVM | 0.760 | 0.036 | 0.330 | 0.462 | 0.747 | 0.469 |
| Standard Transformer | 0.653 | 0.034 | 0.287 | 0.402 | 0.652 | 0.400 |
| 1D-CNN | 0.630 | 0.042 | 0.300 | 0.419 | 0.646 | 0.454 |

Interpretation: MST gives the best average SHERLOC accuracy, weighted-F1, all-label macro-F1, and present-label macro-F1 under the pooled in-situ random-split protocol. The relatively large SD reflects the small and imbalanced SHERLOC label distribution.

## Key Confidence Thresholds

The table below pools held-out predictions from the three random splits, giving 438 validation predictions per model.

| Model | Threshold | Coverage | Accuracy on accepted | Present-label macro-F1 on accepted | False discovery rate among accepted |
|---|---:|---:|---:|---:|---:|
| MST | 0.0 | 1.000 | 0.767 | 0.623 | 0.233 |
| MST | 0.5 | 0.861 | 0.817 | 0.554 | 0.183 |
| MST | 0.7 | 0.546 | 0.904 | 0.785 | 0.096 |
| MST | 0.8 | 0.466 | 0.941 | 0.868 | 0.059 |
| MST | 0.9 | 0.292 | 0.961 | 0.867 | 0.039 |
| PCA-SVM | 0.0 | 1.000 | 0.760 | 0.467 | 0.240 |
| PCA-SVM | 0.8 | 0.463 | 0.931 | 0.466 | 0.069 |
| PLS-DA | 0.0 | 1.000 | 0.765 | 0.452 | 0.235 |
| PLS-DA | 0.8 | 0.320 | 0.993 | 0.700 | 0.007 |
| Standard Transformer | 0.0 | 1.000 | 0.653 | 0.413 | 0.347 |
| Standard Transformer | 0.8 | 0.002 | 1.000 | 1.000 | 0.000 |
| 1D-CNN | 0.0 | 1.000 | 0.630 | 0.428 | 0.370 |
| 1D-CNN | 0.8 | 0.189 | 0.892 | 0.422 | 0.108 |

The full threshold sweep is saved in `sherloc_in_situ_confidence_threshold_sweep.csv`, and the compact threshold table is saved in `sherloc_in_situ_key_thresholds.csv`.
