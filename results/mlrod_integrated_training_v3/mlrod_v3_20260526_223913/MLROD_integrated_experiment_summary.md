# MLROD-Integrated Raman Experiment Summary

Run directory: `publication_repo/results/mlrod_integrated_training_v3/mlrod_v3_20260526_223913`

## Data Integration

The v3 metadata inventory contains the original curated Raman database plus MLROD single-mineral raw spectra from Berlanga et al. (2022; MLROD DOI: 10.48484/PWRB-R137). The complete v3 metadata table has 76,644 spectra. After excluding SHERLOC fine-tuning rows, SHERLOC domain-adaptation rows, and closed-set excluded labels, 75,861 reference plus MLROD spectra were available for supervised reference-domain experiments.

For the model comparison, a reproducible class-balanced MLROD subset was selected to avoid overwhelming the smaller RRUFF, DUV, and meteorite sources. The selected benchmark contained 11,700 spectra: 7,833 train, 1,934 validation, and 1,933 test spectra.

Selected source-by-split counts:

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| MLROD Raman open dataset | 7,200 | 1,800 | 1,800 |
| RRUFF database | 556 | 117 | 112 |
| Lab-acquired DUV spectra | 72 | 16 | 20 |
| Martian meteorite spectra | 5 | 1 | 1 |

All spectra were aligned to a common 0-4000 cm-1 Raman-shift grid with 4,100 points. MLROD wide CSV rows were read directly from their source files by parsing numeric Raman-shift column headers. Regions outside a spectrum's original coverage were zeroed and masked.

## Reference Plus MLROD Model Comparison

Model selection used validation macro-F1. Training augmentation was applied only to the training split; validation and test spectra were measured/non-augmented spectra.

| Model | Best parameters | Val macro-F1 | Combined test acc. | Combined test macro-F1 | Curated reference test acc. | Curated reference macro-F1 | MLROD test acc. | MLROD test macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Standard Transformer | lr=1e-4, d_model=96, layers=3, patch=8 | 0.912 | 0.974 | 0.771 | 0.737 | 0.601 | 0.992 | 0.893 |
| MST | lr=1e-4, d_model=96, layers=3, patch=8 | 0.940 | 0.973 | 0.768 | 0.752 | 0.618 | 0.989 | 0.989 |
| PCA-SVM | stride=8, PCA=120, C=10, gamma=scale | 0.874 | 0.949 | 0.736 | 0.699 | 0.612 | 0.968 | 0.871 |
| 1D-CNN | lr=1e-3, dropout=0.25 | 0.868 | 0.959 | 0.713 | 0.594 | 0.481 | 0.986 | 0.986 |
| PLS-DA | stride=16, components=12 | 0.603 | 0.811 | 0.578 | 0.436 | 0.302 | 0.839 | 0.830 |

Interpretation: the standard Transformer has the highest combined hard-classification accuracy by a small margin, but MST has the highest validation macro-F1, higher curated-reference macro-F1, and the strongest MLROD macro-F1. This supports presenting MST as the more balanced model across minority classes and heterogeneous spectral domains rather than claiming a universal accuracy advantage.

## SHERLOC Fine-Tuning From MLROD-Integrated MST

The best MLROD-integrated MST checkpoint was fine-tuned on the pooled labeled SHERLOC in-situ spectra. The SHERLOC pool contained 730 spectra, split into 584 fine-tuning spectra and 146 held-out validation spectra. Only the last Transformer block, normalization layer, and classification head were updated.

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Zero-shot MLROD/reference-trained MST on held-out SHERLOC | 0.726 | 0.299 |
| Fine-tuned MST on held-out SHERLOC | 0.815 | 0.646 |

Confidence-threshold behavior for the fine-tuned MST on held-out SHERLOC:

| Confidence threshold | Coverage | Accepted spectra | Accuracy on accepted | Macro-F1 on accepted |
|---:|---:|---:|---:|---:|
| 0.0 | 1.000 | 146 | 0.815 | 0.646 |
| 0.5 | 0.952 | 139 | 0.835 | 0.777 |
| 0.7 | 0.897 | 131 | 0.855 | 0.792 |
| 0.8 | 0.808 | 118 | 0.907 | 0.846 |
| 0.9 | 0.658 | 96 | 0.938 | 0.850 |

## Output Files

- `selected_reference_plus_mlrod_samples.csv`: exact spectra used in the benchmark.
- `full_available_source_by_split.csv` and `full_available_class_by_split.csv`: complete available data overview before MLROD subsampling.
- `selected_class_by_split.csv` and `selected_source_by_split.csv`: actual benchmark composition.
- `chemometric/chemometric_hyperparameter_trials.csv`: PCA-SVM and PLS-DA tuning grid.
- `torch_hyperparameter_trials.csv`: CNN, standard Transformer, and MST tuning grid.
- `selected_model_test_summary.csv`: final selected model comparison.
- `selected_model_reports/*.threshold_sweep.csv`: confidence-threshold sweeps for selected reference-domain models.
- `sherloc_pooled_finetune/*threshold_sweep.csv`: zero-shot and fine-tuned MST SHERLOC threshold sweeps.
