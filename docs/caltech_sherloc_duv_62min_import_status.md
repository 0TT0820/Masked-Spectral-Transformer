# Caltech/JPL SHERLOC DUV 62-Mineral Library Import Status

## Source

The source record is:

- Hollis, J. R., Abbey, W., Beegle, L. W., Bhartia, R., Ehlmann, B. L., Miura, J., Monacelli, B., Moore, K., Nordman, A., Scheller, E., Uckert, K., & Wu, Y.-H. (2021). *A deep-ultraviolet Raman and Fluorescence spectral library of 62 minerals for the SHERLOC instrument onboard Mars 2020*. Planetary and Space Science, 209, 105356. DOI: 10.1016/j.pss.2021.105356.
- CaltechAUTHORS record: https://authors.library.caltech.edu/records/qzvym-vsj04

The supplementary file downloaded for local inspection is:

```text
data/external_caltech_sherloc_duv_62min/1-s2.0-S0032063321001951-mmc1.docx
```

## Import Decision

This source is scientifically well matched to the DUV reference-training layer because it reports SHERLOC-analog spectra measured at 248.6 nm for 92 samples representing 62 mineral species. However, the currently available supplementary file is not a machine-readable spectral dataset. Local inspection of the DOCX package found no embedded CSV, XLSX, or OLE spreadsheet objects. The file contains 105 embedded images and text/caption content, including plotted spectra and a calibration peak-position table, but not complete wavenumber-intensity arrays.

For that reason, this source has **not** been merged into the supervised training metadata. Adding plot images or manually digitized curves as if they were raw spectra would create a provenance and reproducibility problem.

## Current Status

| Item | Status |
|---|---|
| Source identified | Complete |
| Supplement downloaded | Complete |
| License checked at source-record level | Complete; reuse is restricted/non-commercial/no-derivatives on the CaltechAUTHORS record |
| Machine-readable spectra found | Not found in the public DOCX supplement |
| Added to supervised DUV training set | No |
| Recommended current use | Citation/source candidate only; request numeric spectra or obtain permission before training import |

## Required Next Step

To include this source in the DUV reference training set, obtain complete numeric spectra from one of the following routes:

1. Author-provided wavenumber-intensity tables with permission to use in this project.
2. A public repository containing the raw numeric spectra under terms compatible with analysis and redistribution.
3. A clearly documented, permission-compatible digitization workflow, retained as a derived-data process rather than treated as raw spectra.

Until one of these conditions is met, the correct manuscript wording is that the Caltech/JPL SHERLOC DUV mineral library was identified as a high-priority DUV reference source, but was not used as supervised training data because complete numeric spectra were not available in the public supplementary file.
