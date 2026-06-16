# Reviewer 1 Comment 1: Spectrum Provenance Supplement

This document describes the reviewer-facing provenance workbook generated from the final v3 compact metadata.
Metadata source: `data\metadata\metadata_training_database_v3_compact.csv`.

The workbook uses P1-P8 sheet labels to avoid conflict with the manuscript's numbered supporting-information tables.
Unavailable fields are encoded as `not reported in source record`; no supplier names, product IDs, or measurement conditions are invented.

Main output workbook:
- `data\metadata\reviewer1_provenance_supplement\Reviewer1_Comment1_Spectrum_Provenance_Supplement_v3.xlsx`

CSV exports:
- `data\metadata\reviewer1_provenance_supplement\v3_P1_full_spectrum_provenance_inventory.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P2_source_task_summary.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P3_source_by_class.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P4_rruff_official_metadata_qc.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P5_quality_control_summary.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P6_split_by_source_and_class.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P7_training_only_augmentation_summary.csv`
- `data\metadata\reviewer1_provenance_supplement\v3_P8_field_dictionary.csv`

Reviewer-facing use:
- P1 is the complete per-spectrum provenance inventory.
- P2-P3 summarize source roles and source-by-mineral-class counts.
- P4 lists RRUFF official identifiers, URLs, excitation wavelength, status, and QC decisions.
- P5-P6 summarize quality control and final train/validation/test/fine-tuning split composition.
- P7 documents training-only materialized augmentation; validation and test spectra remain measured spectra.