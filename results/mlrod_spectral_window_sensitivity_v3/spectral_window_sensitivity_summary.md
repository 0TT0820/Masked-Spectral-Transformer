# Spectral-Window Sensitivity Analysis

This analysis evaluates whether the MLROD-integrated model comparison is driven by the choice of Raman-shift grid. MLROD raw Raman spectra mostly cover approximately 95-1801 cm-1, whereas SHERLOC in-situ Raman spectra are effectively used from about 800 cm-1 upward. Three windows were therefore compared:

- `0-4000 cm-1`: the full manuscript-compatible grid used for heterogeneous reference and SHERLOC workflows.
- `100-1800 cm-1`: the broad common window for MLROD and most non-SHERLOC reference spectra.
- `800-1800 cm-1`: a conservative overlap window closer to the SHERLOC lower Raman limit.

For all windows, spectra were interpolated onto a 1 cm-1 grid, values outside each spectrum's original coverage were not extrapolated, and out-of-coverage regions were zeroed and masked. The same selected reference+MLROD samples, model families, and hyperparameter search protocol were used.

## Reference Plus MLROD Results

| Window | Best validation macro-F1 model | Best validation macro-F1 | Best combined accuracy model | Best combined accuracy | Best combined macro-F1 model | Best combined macro-F1 | MST validation macro-F1 | MST combined accuracy | MST combined macro-F1 |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| 0-4000 cm-1 | MST | 0.940 | Standard Transformer | 0.974 | Standard Transformer | 0.771 | 0.940 | 0.973 | 0.768 |
| 100-1800 cm-1 | MST | 0.937 | MST | 0.976 | PCA-SVM | 0.814 | 0.937 | 0.976 | 0.770 |
| 800-1800 cm-1 | MST | 0.927 | MST | 0.964 | 1D-CNN | 0.820 | 0.927 | 0.964 | 0.753 |

The MST remains the top model by validation macro-F1 in all three spectral windows. This means that the selected MST configuration is not an artifact of the 0-4000 cm-1 grid or of zero-filled regions outside the MLROD coverage. The model ranking by hard test accuracy and combined macro-F1 is more window-dependent: the standard Transformer is slightly higher in full-grid accuracy, PCA-SVM is strongest in 100-1800 cm-1 combined macro-F1, and 1D-CNN is strongest in 800-1800 cm-1 combined macro-F1. This should be reported as a sensitivity result rather than hidden, because it shows that reduced spectral windows change the information available to each model family.

## SHERLOC Fine-Tuning Sensitivity

| Base-training window | Zero-shot SHERLOC accuracy | Zero-shot SHERLOC macro-F1 | Fine-tuned SHERLOC accuracy | Fine-tuned SHERLOC macro-F1 |
|---|---:|---:|---:|---:|
| 0-4000 cm-1 | 0.726 | 0.299 | 0.815 | 0.646 |
| 100-1800 cm-1 | 0.699 | 0.272 | 0.842 | 0.794 |
| 800-1800 cm-1 | 0.452 | 0.082 | 0.897 | 0.864 |

The narrower windows reduce zero-shot transfer, especially for 800-1800 cm-1, because much of the reference-domain spectral context is removed. After SHERLOC fine-tuning, however, the narrower overlap windows improve held-out SHERLOC performance. This indicates that the restricted window reduces cross-domain spectral-range artifacts and focuses the fine-tuned model on the region shared by SHERLOC and the external Raman references.

## Recommended Manuscript Interpretation

The main manuscript should not state that all spectra were simply resampled to 0-4000 cm-1 without qualification. A more accurate statement is that all spectra were mapped to a common grid without extrapolation; outside-coverage values were zeroed and masked. The spectral-window sensitivity analysis should be added to the Supporting Information to show that the conclusions do not depend solely on the full-grid representation. For SHERLOC-specific transfer, the 800-1800 cm-1 overlap-window result can be discussed as a conservative test demonstrating that fine-tuning benefits from a shared spectral window.

## Output Locations

- Full-grid reference run: `publication_repo/results/mlrod_integrated_training_v3/mlrod_v3_20260526_223913`
- 100-1800 cm-1 run: `publication_repo/results/mlrod_spectral_window_sensitivity_v3/mlrod_v3_grid100-1800_n1701_20260527_103101`
- 800-1800 cm-1 run: `publication_repo/results/mlrod_spectral_window_sensitivity_v3/mlrod_v3_grid800-1800_n1001_20260527_104956`
