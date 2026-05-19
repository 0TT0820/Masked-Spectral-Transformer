# SHERLOC SaU 008 Calibration Target Data

This directory adds PDS-traceable SHERLOC spectra from the Mars meteorite
SaU 008 calibration target. The products were selected from the official Mars
2020 SHERLOC processed-data inventory using RRS product identifiers with
`SRLC15030` or `SRLC15031`.

Scientific-use note: these PDS RRS products identify the bulk calibration
target as Mars meteorite SaU 008, but they do not assign point-level mineral
labels. For this reason, the spectra are marked as `supervised_training_usable =
False` and `domain_adaptation_usable = True`. They should be used for
SHERLOC-domain adaptation, calibration-domain checking, or later manual
mineral-label review, rather than as closed-set mineral-category labels.

## Provenance Basis

- PDS processed collection DOI: `10.17189/1522643`
- SHERLOC User Guide: external calibration targets use `SRLC15*`; target 3 is
  Mars Meteorite SaU 008.
- SHERLOC RDR SIS: calibration target material number 3 is Mars Meteorite SaU
  008.

## Generated Files

- `data/metadata/metadata_sherloc_sau008_calibration_mean_spectra.csv`
- `data/metadata/metadata_sherloc_sau008_calibration_point_index.csv`
- `data/overview/sherloc_sau008_calibration/sherloc_sau008_pds_products.csv`
- `data/overview/sherloc_sau008_calibration/sherloc_sau008_summary.csv`
- `data/sherloc_sau008_calibration/mean_spectra/*.csv`

## Current Counts

- PDS RRS products: 12
- Mean region spectra extracted: 36
- Point spectra indexed in source RRS files: 23655

The point-level intensities remain in the original downloaded PDS RRS CSV files
under `data/external_sherloc_sau008_pds/products/`. The point index table
records the source product, detector region, and row index needed to recover
each spectrum exactly.
