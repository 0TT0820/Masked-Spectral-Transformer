# Final v3 Data, Results, And Figure Index

This document is the file-level guide for the current manuscript revision. It
identifies the final data products, model summaries, and figure source files
that should be used for manuscript checking and GitHub release packaging.

## Final Dataset

| Purpose | File |
|---|---|
| Compact public metadata inventory | `data/metadata/metadata_training_database_v3_compact.csv` |
| Large regenerated integrated metadata | `data/metadata/metadata_training_database_v3_mlrod_integrated.csv` |
| Large integrated metadata with meteorite spot supplement | `data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv` |
| Spectrum-level provenance workbook | `data/metadata/reviewer1_provenance_supplement/Reviewer1_Comment1_Spectrum_Provenance_Supplement_v3.xlsx` |
| SHERLOC PE/PH-aligned fine-tuning pool | `data/metadata/metadata_sherloc_finetune_pool_comprehensive_v2_peph_class.csv` |

The compact metadata table now contains 77,444 spectrum-level records: MLROD
Raman open dataset (74,961), synthetic meteorite mineral spot spectra (800),
RRUFF (791), SHERLOC in-situ Mars 2020 (730), laboratory DUV spectra (119),
SHERLOC SaU 008 calibration-target spectra (36), and Martian meteorite spectra
(7). The 800 spot rows are training-only and retain the source flag
`synthetic_teaching_only`; they should not be described as measured meteorite
spectra. The large integrated MLROD metadata files are regenerated locally and
are intentionally ignored by Git.

## Final Model And Numerical Results

The authoritative numerical summary is:

```text
results/band_aware_mlrod_v3/band_aware_model_selection_summary.md
```

The selected manuscript MST is:

```text
band_multiscale_mst_patch8_d128_ce
```

The principal reported metrics are:

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Reference+MLROD measured test spectra | 0.976 | 0.837 |
| SHERLOC zero-shot validation | 0.493 | 0.161 |
| SHERLOC fine-tuned validation | 0.863 | 0.820 |

The full 0-4000 cm-1 run directory is:

```text
results/band_aware_mlrod_v3/band_aware_mlrod_v3_grid0-4000_n4001_20260527_192236/
```

Run-level CSV files may include validation-best sensitivity candidates. The
manuscript configuration and interpretation should follow
`band_aware_model_selection_summary.md`.

## Meteorite Spot Supplement Sensitivity Result

The training-only spot supplement run is summarized in:

```text
results/band_aware_mlrod_v3_spots/meteorite_spot_supplement_summary.md
```

The main MST run with the 800 training-only spot rows selected 8,633 train,
1,934 validation, and 1,933 test spectra. It achieved 0.976 accuracy / 0.785
macro-F1 on the measured Reference+MLROD test set, 0.726 / 0.240 for SHERLOC
zero-shot validation, and 0.877 / 0.693 after SHERLOC fine-tuning. Because the
spot file is flagged as `synthetic_teaching_only`, this result is a transparent
sensitivity analysis rather than a replacement for the measured-data final
manuscript metrics above.

## Final Figure 5-7 Products

| Manuscript purpose | Data or example directory |
|---|---|
| Model-performance bars and tables | `figures/origin_model_performance_data/` |
| Class-wise performance panels | `figures/origin_classwise_performance_data/` |
| Example Figure 5 layouts | `figures/model_performance_examples/` |
| Example class-wise layouts | `figures/classwise_performance_examples/` |
| Final Figure 7 panel assembly | `figures/fig7_panels/` |

These folders contain figure-ready CSV tables and example SVG/PNG layouts for
the revised performance, per-class, and confidence-threshold panels.

## Final Interpretability Figure

The final SHERLOC confusion-case SHAP panel is:

```text
results/shap_confusion_explanations_v4_final/figure_sherloc_confusion_peak_shap_publication_style.svg
results/shap_confusion_explanations_v4_final/figure_sherloc_confusion_peak_shap_publication_style.png
results/shap_confusion_explanations_v4_final/figure_sherloc_confusion_peak_shap_publication_style.pdf
```

The source tables are:

```text
results/shap_confusion_explanations_v4_final/sherloc_confusion_peak_shap_values.csv
results/shap_confusion_explanations_v4_final/combined_confusion_peak_shap_values.csv
results/shap_confusion_explanations_v4_final/reference_representative_peak_shap_values.csv
```

This figure is a qualitative band-level explanation of difficult SHERLOC cases.
It should be discussed as interpretability evidence, not as an additional
accuracy metric.

## Final Embedding Figure

The final three-panel t-SNE figure is:

```text
results/tsne_embedding_figures_v1/figure_mst_tsne_three_panel_combined.svg
results/tsne_embedding_figures_v1/figure_mst_tsne_three_panel_combined.png
results/tsne_embedding_figures_v1/figure_mst_tsne_three_panel_combined.pdf
```

Its source tables are:

```text
results/tsne_embedding_figures_v1/reference_test_mst_tsne_embedding_points.csv
results/tsne_embedding_figures_v1/sherloc_pooled_finetuned_mst_tsne_embedding_points.csv
results/tsne_embedding_figures_v1/sherloc_validation_finetuned_mst_tsne_embedding_points.csv
results/tsne_embedding_figures_v1/tsne_embedding_figure_summary.csv
```

The t-SNE panels visualize learned MST feature embeddings for the measured
reference/MLROD test set, all pooled labeled SHERLOC spectra, and the held-out
SHERLOC validation subset. They are qualitative diagnostics only.

## Main Regeneration Commands

```bash
python src/download_mlrod_raw_raman.py
python src/build_mlrod_integrated_dataset_v3.py \
  --out-metadata data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv
python src/build_compact_v3_metadata.py \
  --full-metadata data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv

python src/run_band_aware_mlrod_v3_experiments.py \
  --metadata-file data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv \
  --out-dir results/band_aware_mlrod_v3_spots \
  --grid-min 0 --grid-max 4000 --grid-points 4001 \
  --trial-set main \
  --main-model band_multiscale_mst_patch8_d128_ce \
  --run-sherloc

python src/run_peak_confusion_explanations.py --out-dir results/shap_confusion_explanations_v4_final
python src/plot_sherloc_confusion_publication_style.py
python src/plot_mst_tsne_embeddings.py
python src/plot_mst_tsne_combined.py
```

## Legacy And Exploratory Outputs

Folders with names containing `v2`, `review_updated_training`, `graph_embedding`,
or `lda_embedding` are retained as audit or exploratory products. They are not
the final manuscript source unless a future revision explicitly promotes them.
