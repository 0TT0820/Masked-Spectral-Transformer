# Raman Spectral Quality Control v2

This QC layer was added to address reviewer concerns about spectral provenance,
RRUFF quality heterogeneity, fluorescence-dominated spectra, weak Raman bands,
and the physical unsuitability of halides as a closed-set Raman class.

The original all-source metadata table is retained unchanged. The filtered table
`data/metadata/metadata_training_database_v2_qc_filtered.csv` adds numerical
QC metrics and marks spectra excluded from the main closed-set supervised
comparison as `reference_qc_excluded`.

## QC Rules

1. Hard exclusions: unreadable files, fewer than 50 finite points, spectral span
   below 300 cm-1, flat/noise-dominated spectra with no detected Raman bands,
   and halides.
2. Manual-review flags: RRUFF records with low-confidence identification status,
   unmatched RRUFF header metadata, high spike/discontinuity fraction, or spectra
   dominated by broad continuum with too few Raman bands.
3. SHERLOC spectra are retained unless they fail hard numerical checks, because
   rover DUV spectra are expected to be noisier than laboratory references.
4. The main model-comparison subset uses only spectra marked
   `main_training_keep`.

## Summary

- Reference candidate spectra before QC: 917
- Reference spectra after hard exclusions only: 900
- Reference spectra after balanced QC policy: 794
- Reference spectra after strict QC policy: 637

The balanced policy is recommended for the main reviewer experiment because it
removes physically problematic classes and low-confidence/continuum-dominated
reference spectra while avoiding an overly aggressive automatic exclusion of
spectra that only show spike-like discontinuities. Those spike-flagged spectra
remain visible in the spectrum-level QC table and should be inspected manually
before final archival release.

### Decisions

| decision                       |   n_spectra |
|:-------------------------------|------------:|
| main_training_keep             |        1403 |
| manual_review_keep_in_metadata |         263 |
| exclude_from_closed_set        |          17 |

### Decisions by Source

| source_type_normalized                            | raman_qc_decision              |   n_spectra |
|:--------------------------------------------------|:-------------------------------|------------:|
| Lab-acquired DUV spectra                          | exclude_from_closed_set        |          11 |
| Lab-acquired DUV spectra                          | main_training_keep             |         104 |
| Lab-acquired DUV spectra                          | manual_review_keep_in_metadata |           4 |
| Martian meteorite spectra                         | main_training_keep             |           7 |
| RRUFF database                                    | exclude_from_closed_set        |           6 |
| RRUFF database                                    | main_training_keep             |         526 |
| RRUFF database                                    | manual_review_keep_in_metadata |         259 |
| SHERLOC calibration target Mars meteorite SaU 008 | main_training_keep             |          36 |
| SHERLOC in-situ Mars 2020                         | main_training_keep             |         730 |
