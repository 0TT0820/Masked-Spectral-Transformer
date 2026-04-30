# Data Guide

## Parent Dataset

The parent dataset contains 945 Raman spectra. Each spectrum has a row in:

```text
data/metadata/metadata_parent_945.csv
```

The parent spectra are stored as individual CSV files in:

```text
data/spectra/parent/
```

Each spectrum file contains two columns: wavenumber and intensity.

RRUFF spectra include additional official header fields parsed from the downloaded RRUFF text spectra:

```text
data/metadata/rruff_official_header_metadata.csv
data/metadata/metadata_parent_945_rruff_enriched.csv
```

These fields include the official RRUFF identifier, mineral name, ideal chemistry, measured chemistry where available, locality, source collection, owner, sample description, identification status, and official RRUFF URL. The extraction script is `src/enrich_metadata_from_rruff_headers.py`.

## Metadata Fields

Key columns in `metadata_parent_945.csv` include:

- `spectrum_id`: stable identifier for each parent spectrum.
- `file_path`: repository-relative path to the spectrum CSV file.
- `file_name_clean`: original spectrum filename.
- `major_category`: original mineral superclass label.
- `subtype_label`: mineral species or subtype label where available.
- `mineral_species`: parsed species name.
- `source_type`: data source category.
- `source_id`: source-specific record or sample identifier.
- `excitation_nm`: excitation wavelength.
- `instrument`: instrument or source database information.
- `sample_provenance`: provenance notes.
- `measurement_conditions`: measurement-condition notes.
- `label_basis`: basis for the label assignment.
- `reference`: literature or database reference.
- `parent_group`: group identifier used to avoid leakage.
- `split_main`: group-wise train/validation/test/external split.
- `qc_status`: quality-control flag.
- `qc_reason`: reason for review or exclusion.
- `rruff_official_name`: official RRUFF mineral name for RRUFF-derived spectra.
- `rruff_ideal_chemistry`: official ideal chemistry from the RRUFF header.
- `rruff_measured_chemistry`: measured chemistry from the RRUFF header where available.
- `rruff_locality`: RRUFF specimen locality.
- `rruff_source`: RRUFF source collection.
- `rruff_status`: RRUFF identification status.
- `rruff_url`: official RRUFF record URL.

## Source Types

The source inventory is summarized in:

```text
data/overview/parent_by_source_type.csv
```

Current source counts:

- RRUFF database: 791 spectra
- Laboratory-acquired DUV spectra: 119 spectra
- SHERLOC in-situ spectra: 31 spectra
- Martian meteorite spectra: 4 spectra

## Split Policy

The split file is:

```text
data/metadata/metadata_parent_group_split.csv
```

The split is group-wise. Spectra sharing the same `parent_group`, such as different excitation wavelengths from the same RRUFF record, are not split across train, validation, and test sets.

For reviewer-facing reporting, use `data/overview/review_data_inventory/`. The key files are:

```text
data/overview/review_data_inventory/dataset_stage_summary.csv
data/overview/review_data_inventory/dataset_flow_by_review_class.csv
data/overview/review_data_inventory/source_split_class_matrix.csv
data/overview/review_data_inventory/sherloc_product_group_summary.csv
data/overview/review_data_inventory/meteorite_spot_inventory.csv
data/overview/review_data_inventory/spectrum_level_provenance_review.csv
```

This inventory separates raw parent spectra from review-ready Earth-domain train/validation/test spectra, reproducible augmentation targets, SHERLOC external/candidate transfer groups, and excluded halide spectra.

SHERLOC in-situ spectra are marked as `external_sherloc` and are not mixed into the Earth-domain training split.

## Quality-Control Notes

The QC review table is:

```text
data/metadata/metadata_parent_qc_review.csv
```

Halide and weak-feature spectra are flagged because pure crystalline halite and related salts may be Raman-inactive or weak under conventional conditions. Clay, mica, and serpentine records are flagged because their taxonomy overlaps within phyllosilicates and should be harmonized before final supervised training.

## Data-Source Overview Tables

Publication-ready summary tables are in:

```text
data/overview/
```

Important files:

- `parent_by_source_and_category.csv`
- `parent_by_excitation_and_source.csv`
- `parent_by_split_and_category.csv`
- `parent_provenance_inventory.csv`
- `augmented_by_lineage_status.csv`
- `reviewer_data_transparency_checklist.csv`
