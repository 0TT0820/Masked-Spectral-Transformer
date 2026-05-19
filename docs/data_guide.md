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

Current parent source counts:

- RRUFF database: 791 spectra
- Laboratory-acquired DUV spectra: 119 spectra
- SHERLOC in-situ spectra: 31 spectra
- Martian meteorite spectra: 4 spectra

After adding the SHERLOC point-level fine-tuning dataset, traceable Martian
meteorite supplements, and SHERLOC SaU 008 DUV calibration-target spectra, the
current all-source training database is:

```text
data/metadata/metadata_training_database_v2_all_sources.csv
```

Current all-source metadata rows in that table:

- RRUFF database: 791 spectra
- SHERLOC in-situ Mars 2020: 730 spectra
- Laboratory-acquired DUV spectra: 119 spectra
- Martian meteorite spectra: 7 spectra
- SHERLOC calibration target Mars meteorite SaU 008: 36 spectra

The current DUV-only spectral library is:

```text
data/metadata/metadata_duv_training_library_v1.csv
```

It contains 885 DUV spectra: 849 rows with supervised labels and 36 SaU 008
calibration-target rows reserved for unlabeled domain adaptation/manual review.
The key role columns are `source_domain`, `training_role`,
`supervised_label_usable_v2`, `duv_library_include`, and `split_v2`.

Summary tables are stored in:

```text
data/overview/training_database_v2/
```

## Split Policy

The split file is:

```text
data/metadata/metadata_parent_group_split.csv
```

The split is group-wise. Spectra sharing the same `parent_group`, such as different excitation wavelengths from the same RRUFF record, are not split across train, validation, and test sets.

For detailed data-flow reporting, use `data/overview/data_inventory/`. The key files are:

```text
data/overview/data_inventory/dataset_stage_summary.csv
data/overview/data_inventory/dataset_flow_by_class.csv
data/overview/data_inventory/source_split_class_matrix.csv
data/overview/data_inventory/sherloc_product_group_summary.csv
data/overview/data_inventory/meteorite_spot_inventory.csv
data/overview/data_inventory/spectrum_level_provenance.csv
```

This inventory separates raw parent spectra from Earth-domain train/validation/test spectra, reproducible augmentation targets, SHERLOC external/candidate transfer groups, and excluded halide spectra.

SHERLOC in-situ spectra are marked as `external_sherloc` and are not mixed into the Earth-domain training split.

## SHERLOC Region Fine-Tuning Dataset

This release adds a SHERLOC region dataset derived from labeled point-level
mineral assignments in Dourbes, Garde/Bellegarde, Guillaumes, and Quartier.
Only points with explicit mineral labels in the region spreadsheets are used;
unlabeled points are treated as noise/background and are not included as
training labels.

The training-ready combined table is:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv
```

Supporting files are:

```text
data/metadata/metadata_sherloc_region_points_only.csv
data/metadata/sherloc_region_detail_to_ss_mapping.csv
data/metadata/sherloc_pds_product_provenance.csv
data/metadata/sherloc_pds_product_provenance_readme.md
data/metadata/sherloc_spectrum_to_pds_crosswalk.csv
data/metadata/sherloc_region_point_extraction_manifest.csv
data/metadata/sherloc_region_table1_training_summary.csv
data/overview/sherloc_regions/
```

The mapping file records the relation between each region/detail sheet and the
corresponding `ss__...csv` SHERLOC spectral product. The extraction manifest
records the point column, mineral label, mapped manuscript Table 1 superclass,
and source file for each usable point-level spectrum. If one point has two
accepted mineral labels, it is represented as two labeled records, preserving
the same spectrum path but separate label rows.

These SHERLOC spectra are intended for in-situ adaptation and independent
region/target transfer experiments, not for generating synthetic augmentation.

## SHERLOC SaU 008 Calibration-Target DUV Dataset

SHERLOC calibration observations of the Mars meteorite SaU 008 target are
included as real SHERLOC DUV spectra in the DUV library:

```text
data/metadata/metadata_sherloc_sau008_calibration_mean_spectra.csv
data/metadata/metadata_sherloc_sau008_calibration_point_index.csv
data/sherloc_sau008_calibration/mean_spectra/
data/overview/sherloc_sau008_calibration/
```

These rows are PDS-traceable and useful for domain adaptation, instrument-domain
inspection, and calibration-target review. They are not used as closed-set
supervised mineral labels because the PDS RRS products do not provide
point-level mineral assignments for the extracted spectra.

The processing script is:

```text
src/build_sherloc_sau008_calibration_dataset.py
```

The consolidated all-source and DUV-library tables can be regenerated with:

```text
src/build_training_database_v2.py
```

### Montpezat and Alfalfa Candidate Spectra

Montpezat and Alfalfa are documented separately because the public paper gives
scan-level mineral detections but does not provide the point-level label
spreadsheet used above for Dourbes, Garde/Bellegarde, Guillaumes, and Quartier.
Their PDS-traceable candidate products are recorded in:

```text
data/metadata/sherloc_montpezat_alfalfa_pds_products.csv
data/metadata/metadata_sherloc_montpezat_alfalfa_weak_candidates.csv
data/metadata/sherloc_montpezat_alfalfa_candidate_summary.csv
data/metadata/sherloc_montpezat_alfalfa_candidate_readme.md
data/sherloc_montpezat_alfalfa_candidates/
```

The candidate rows are appended only to the reference table:

```text
data/metadata/metadata_parent_945_plus_sherloc_regions_with_montpezat_alfalfa_candidates.csv
```

They are marked `sherloc_training_label_usable=False` because their labels are
scan-level weak labels, not point-level mineral assignments. They should not be
used in the default closed-set fine-tuning experiment unless point-level labels
are curated later.

## Martian Meteorite Mendeley Supplement

Additional public Martian meteorite Raman data were screened from Mendeley
Data. Only records with explicit mineral labels are added to the supervised
training-ready metadata:

```text
data/metadata/metadata_martian_meteorite_mendeley_supervised_supplement.csv
data/metadata/metadata_training_ready_plus_martian_meteorite_mendeley.csv
data/spectra/martian_meteorite_mendeley/
```

The usable supervised supplement contains three spectra from the CC BY 4.0
dataset DOI `10.17632/c6t3v22x2x.1`: ilmenite, magnetite, and
titanomagnetite. All three are mapped to the manuscript Table 1 superclass
`Oxides/Hydroxides`.

A second Mendeley dataset, DOI `10.17632/97hjg7hcft.1`, contains 11 paired
wavenumber-intensity spectra for the same MIL paired Martian meteorites, but
the downloaded workbook does not provide per-spectrum mineral labels. These
spectra are therefore retained only as non-training candidates:

```text
data/metadata/metadata_martian_meteorite_mendeley_unlabeled_candidates.csv
data/spectra/martian_meteorite_mendeley/unlabeled_candidates_97hjg7hcft/
```

The processing script is:

```text
src/build_martian_meteorite_mendeley_supplement.py
```

## SHERLOC PDS Product Provenance

The SHERLOC products are traced at PDS product level in:

```text
data/metadata/sherloc_pds_product_provenance.csv
data/metadata/sherloc_spectrum_to_pds_crosswalk.csv
```

The table was generated with:

```text
src/build_sherloc_pds_provenance.py
```

Each row of `sherloc_pds_product_provenance.csv` corresponds to one unique Mars
2020 SHERLOC `RRS` processed spectroscopy product. The table includes the PDS
logical identifier, direct CSV and XML label URLs, bundle DOI, sol, SCLK,
site/drive, SRLC sequence, processing flags, observation time from the XML
label, and upstream raw/intermediate PDS logical identifiers.

Each row of `sherloc_spectrum_to_pds_crosswalk.csv` corresponds to one local
SHERLOC point spectrum record. Local point spectra are therefore traceable from
the repository row, through the local `ss__...` filename, to the authoritative
PDS4 XML label.

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
- `data_transparency_checklist.csv` (legacy filename; provenance checklist)
