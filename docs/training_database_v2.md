# Training Database v2

This versioned metadata set reorganizes the Raman training data after adding
PDS-traceable SHERLOC DUV spectra. It keeps a clear distinction between
supervised labels and DUV spectra that are useful only for domain adaptation or
manual review.

## Files

- `data/metadata/metadata_training_database_v2_all_sources.csv`
- `data/metadata/metadata_duv_training_library_v1.csv`
- `data/overview/training_database_v2/all_sources_counts.csv`
- `data/overview/training_database_v2/duv_library_counts.csv`
- `data/overview/training_database_v2/duv_source_by_label.csv`
- `data/overview/training_database_v2/all_source_by_training_role.csv`

## Counts

- All-source metadata rows: 1683
- DUV-library rows: 885
- DUV rows usable for supervised labels: 849
- DUV rows reserved for domain adaptation or manual review: 36

## Interpretation

The DUV library includes laboratory DUV spectra, labeled SHERLOC in-situ Mars
2020 spectra, and SHERLOC SaU 008 calibration-target spectra. SaU 008 spectra
are included because they are real SHERLOC DUV measurements of a Martian
meteorite calibration target, but their PDS products do not provide point-level
mineral labels. They are therefore marked as `duv_domain_adaptation_bulk_unlabeled`
and excluded from closed-set supervised mineral classification unless later
manual point-level labels are added.

The Caltech/JPL SHERLOC-analog 62-mineral library is documented separately in
`docs/caltech_sherloc_duv_62min_import_status.md`. It is not imported into the
numeric training table because the public supplement located so far contains
plotted spectra rather than machine-readable wavenumber-intensity arrays.
