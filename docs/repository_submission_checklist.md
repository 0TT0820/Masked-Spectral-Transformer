# Repository Submission Checklist

This file maps the repository contents to common software-publication review requirements.

## Required Public Repository Items

- Clear license: `LICENSE` for code and `DATA_LICENSE.md` for shared data files.
- English README: `README.md`.
- Dependencies and computational requirements: `requirements.txt`, `environment.yml`, and `docs/reproducibility.md`.
- Reproducible material for main results: `src/run_band_aware_mlrod_v3_experiments.py`, `data/metadata/metadata_training_database_v3_compact.csv`, regenerated `data/metadata/metadata_training_database_v3_mlrod_integrated.csv`, and `results/band_aware_mlrod_v3/band_aware_model_selection_summary.md`.
- Reproducible material for the meteorite spot supplement sensitivity run: `data/minerals_100spots_wide.csv`, regenerated `data/metadata/metadata_training_database_v3_mlrod_integrated_spots.csv`, and `results/band_aware_mlrod_v3_spots/meteorite_spot_supplement_summary.md`. The supplement is explicitly flagged as `synthetic_teaching_only`.
- Test or tutorial workflows: `docs/tutorials/`.
- User guide: `docs/user_guide.md`.
- No archive-only distribution: raw spectra are stored as individual CSV files under `data/spectra/parent/`.
- Data provenance: `data/metadata/reviewer1_provenance_supplement/Reviewer1_Comment1_Spectrum_Provenance_Supplement_v3.xlsx`, `data/metadata/metadata_training_database_v3_compact.csv`, and `docs/reviewer1_comment1_provenance_supplement.md`.
- Detailed data-flow inventory: `data/overview/data_inventory/`, especially `dataset_stage_summary.csv`, `dataset_flow_by_class.csv`, and `spectrum_level_provenance.csv`.
- Augmentation lineage: `src/augment_raman_dataset.py` writes parent identifiers, random seeds, and parameter JSON for every generated spectrum.
- Manuscript figures: final performance figure data under `figures/`, SHAP interpretability outputs under `results/shap_confusion_explanations_v4_final/`, and t-SNE embedding outputs under `results/tsne_embedding_figures_v1/`.
- Current file-level release map: `docs/final_results_and_figures_index.md`.

## Suggested GitHub Metadata

Description:

```text
Physics-informed Raman spectral classification for planetary mineral identification with transparent data provenance and MST/chemometric baselines.
```

Topics:

```text
raman-spectroscopy, planetary-science, mars, sherloc, mineral-classification, transformer, machine-learning, reproducible-research
```
