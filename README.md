# Raman MST Mineral Classification Repository

This repository contains the revised data-processing, model-training, SHERLOC fine-tuning, figure-data, and provenance scripts for the manuscript:

**A Lightweight Physics-Informed Spectral Transformer for On-Board Raman Mineral Identification on Mars**

The current public version corresponds to the final v3 revision workflow. It uses quality-controlled multi-source Raman spectra, materialized training-only augmentation, an expanded MLROD public Raman reference set, and pooled SHERLOC in-situ spectra for Mars-domain fine-tuning/validation.

## Repository Layout

```text
data/
  metadata/                         Curated metadata and provenance tables
  spectra/                          Packaged parent, DUV, meteorite, SHERLOC, and derived spectra
figures/                            Figure-ready CSV files and example SVG/PNG layouts
results/
  band_aware_mlrod_v3/              Final measured-data model-selection and SHERLOC fine-tuning outputs
  band_aware_mlrod_v3_spots/        Training-only meteorite spot supplement sensitivity run
  shap_confusion_explanations_v4_final/
                                     Final peak-aware SHAP interpretability figures and data
  tsne_embedding_figures_v1/         Final MST embedding t-SNE figures and point tables
src/                                Data curation, augmentation, training, evaluation, and plotting scripts
docs/                               Reviewer-facing data dictionaries and provenance notes
```

## Final Dataset

The final compact metadata table is:

```text
data/metadata/metadata_training_database_v3_compact.csv
```

It contains 77,444 spectrum-level records after provenance harmonization and quality-control annotation. Of these, 76,644 rows are measured or mission-derived records, and 800 rows are a transparent training-only meteorite-mineral spot supplement whose source file is marked `synthetic_teaching_only`.

| Source group | N |
|---|---:|
| MLROD Raman open dataset | 74,961 |
| Synthetic meteorite mineral spot spectra | 800 |
| RRUFF database | 791 |
| SHERLOC in-situ Mars 2020 | 730 |
| Lab-acquired DUV spectra | 119 |
| SHERLOC calibration target Mars meteorite SaU 008 | 36 |
| Martian meteorite spectra | 7 |

The full integrated MLROD table is large and is intentionally ignored by Git:

```text
data/metadata/metadata_training_database_v3_mlrod_integrated.csv
data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv
```

Regenerate it with:

```bash
python src/download_mlrod_raw_raman.py
python src/build_mlrod_integrated_dataset_v3.py
```

The spot-supplement large metadata can also be regenerated explicitly with:

```bash
python src/build_mlrod_integrated_dataset_v3.py \
  --out-metadata data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv
python src/build_compact_v3_metadata.py \
  --full-metadata data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv
```

## Final Model

The selected MST variant is:

```text
band_multiscale_mst_patch8_d128_ce
```

Key architectural features:

- Raman-shift physical positional encoding over the 0-4000 cm-1 grid.
- Valid-range mask so missing/unmeasured spectral regions do not contribute to attention or pooling.
- Multi-scale local Raman-band frontend before patch tokenization.
- Band-aware attention pooling over valid encoded tokens.
- Training-only Raman-aware augmentation for minority reference classes.

## Main Results

Final summary file:

```text
results/band_aware_mlrod_v3/band_aware_model_selection_summary.md
```

This summary is the authoritative manuscript-level result table for the final
v3 revision. Some run-level CSV files retain validation-best or sensitivity
candidates for auditability; those should not be confused with the selected
main manuscript configuration unless explicitly identified in the summary file.

Primary reported metrics:

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Reference+MLROD measured test spectra | 0.976 | 0.837 |
| SHERLOC zero-shot validation | 0.493 | 0.161 |
| SHERLOC fine-tuned validation | 0.863 | 0.820 |

Validation macro-F1 is reported for hyperparameter-screening transparency. The manuscript-reported main MST configuration is the multi-scale, band-aware full-grid configuration that best matched the final architecture and gave the highest class-balanced measured-test macro-F1 among the tuned full-grid candidates, while the validation-best patch-only candidate is retained as a sensitivity comparison. SHERLOC reporting includes both zero-shot and fine-tuned protocols. Validation and test spectra are measured spectra only; augmented spectra are used only in the training split.

The current repository also includes a main-MST sensitivity run with the 800
training-only meteorite spot supplement:

```text
results/band_aware_mlrod_v3_spots/meteorite_spot_supplement_summary.md
```

That run used 8,633 train, 1,934 validation, and 1,933 test spectra. It achieved
0.976 accuracy / 0.785 macro-F1 on the Reference+MLROD measured test set,
0.726 / 0.240 for SHERLOC zero-shot validation, and 0.877 / 0.693 after SHERLOC
fine-tuning. Because the spot file is explicitly marked `synthetic_teaching_only`,
this run is reported as a sensitivity/training-supplement result, not as a
replacement for the measured-data manuscript result.

## Reproduce The Final Experiment

Install dependencies:

```bash
pip install -r requirements.txt
```

Rebuild the large integrated MLROD metadata if it is not already present:

```bash
python src/download_mlrod_raw_raman.py
python src/build_mlrod_integrated_dataset_v3.py
```

Then run the final band-aware MST/SHERLOC experiment:

```bash
python src/run_band_aware_mlrod_v3_experiments.py \
  --metadata-file data/metadata/metadata_training_database_v3_mlrod_integrated.csv \
  --out-dir results/band_aware_mlrod_v3 \
  --grid-min 0 --grid-max 4000 --grid-points 4001 \
  --main-model band_multiscale_mst_patch8_d128_ce \
  --run-sherloc
```

The default script parameters already match the final 0-4000 cm-1 workflow.

## Reproduce Final Figures

The final model-performance panels are generated from the figure-ready CSV
tables under `figures/`. The final interpretability and embedding figures are
generated from the reported MST checkpoints and predictions with:

```bash
python src/run_peak_confusion_explanations.py --out-dir results/shap_confusion_explanations_v4_final
python src/plot_sherloc_confusion_publication_style.py
python src/plot_mst_tsne_embeddings.py
python src/plot_mst_tsne_combined.py
```

The resulting publication-style outputs are:

```text
results/shap_confusion_explanations_v4_final/figure_sherloc_confusion_peak_shap_publication_style.svg
results/tsne_embedding_figures_v1/figure_mst_tsne_three_panel_combined.svg
```

The SHAP figure is used as a qualitative, band-level explanation of selected
SHERLOC confusion cases. The t-SNE figure is used only as an embedding-space
visualization of the final MST representation; quantitative claims are based on
the classification and threshold tables, not on t-SNE geometry.

## Provenance Supplement

Build the reviewer-facing spectrum provenance workbook:

```bash
python src/build_reviewer1_provenance_supplement.py
```

Main output:

```text
data/metadata/reviewer1_provenance_supplement/Reviewer1_Comment1_Spectrum_Provenance_Supplement_v3.xlsx
```

The workbook uses P1-P8 sheet labels:

- P1: complete spectrum-level provenance inventory.
- P2-P3: source-role and source-by-class summaries.
- P4: RRUFF official metadata and QC decisions.
- P5-P6: quality-control and final split summaries.
- P7: training-only augmentation summary.
- P8: field dictionary.

## Figure Data

Figure-ready tables and examples are stored under:

```text
figures/origin_model_performance_data/
figures/origin_classwise_performance_data/
figures/model_performance_examples/
figures/classwise_performance_examples/
figures/fig7_panels/
```

These files support the revised Figure 5-7 performance, class-wise, and confidence-threshold panels.

Final interpretability and representation-learning figure products are stored
under:

```text
results/shap_confusion_explanations_v4_final/
results/tsne_embedding_figures_v1/
```

For a single file-by-file guide to the current manuscript data, results, and
figures, see:

```text
docs/final_results_and_figures_index.md
```

## Notes On Large Files

Raw external downloads, model checkpoints, cache folders, and large regenerated metadata products are excluded through `.gitignore`. The repository keeps compact metadata, curated source spectra, scripts, and final result summaries needed to audit and reproduce the manuscript workflow.

Earlier v2 and exploratory folders are retained only as revision history and
sensitivity checks. The final manuscript should cite the v3 metadata, the
`band_aware_mlrod_v3` summary, and the figure outputs listed above.
