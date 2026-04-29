# Repository Submission Checklist

This file maps the repository contents to common software-publication review requirements.

## Required Public Repository Items

- Clear license: `LICENSE` for code and `DATA_LICENSE.md` for shared data files.
- English README: `README.md`.
- Dependencies and computational requirements: `requirements.txt`, `environment.yml`, and `docs/reproducibility.md`.
- Reproducible material for main results: `src/train_review_comparison.py`, `data/metadata/`, `data/spectra/parent/`, and `results/model_comparison/`.
- Test or tutorial workflows: `docs/tutorials/`.
- User guide: `docs/user_guide.md`.
- No archive-only distribution: raw spectra are stored as individual CSV files under `data/spectra/parent/`.
- Data provenance: `data/metadata/metadata_parent_945.csv`, `data/metadata/repository_spectra_index.csv`, and `data/overview/parent_provenance_inventory.csv`.
- Reviewer data-flow inventory: `data/overview/review_data_inventory/`, especially `dataset_stage_summary.csv`, `dataset_flow_by_review_class.csv`, and `spectrum_level_provenance_review.csv`.
- Augmentation lineage: `src/augment_raman_dataset.py` writes parent identifiers, random seeds, and parameter JSON for every generated spectrum.
- Manuscript figures: `assets/figures/` with an overview in `docs/figure_gallery.md`.

## Suggested GitHub Metadata

Description:

```text
Physics-informed Raman spectral classification for planetary mineral identification with transparent data provenance and MST/chemometric baselines.
```

Topics:

```text
raman-spectroscopy, planetary-science, mars, sherloc, mineral-classification, transformer, machine-learning, reproducible-research
```
