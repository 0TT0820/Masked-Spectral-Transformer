# MLROD Berlanga et al. (2022) Data Intake and Overlap Report

Source paper: Berlanga et al. (2022), *Earth and Space Science*, DOI: 10.1029/2021EA002125.

Source dataset: MLROD: Machine Learning Raman Open Dataset, DOI: 10.48484/PWRB-R137.

Local status:

- ODR/AHED metadata has been harvested into `mlrod_sample_catalog.csv` and `mlrod_file_catalog.csv`.
- Average Raman spectra for 15 training sample groups have been downloaded into `../average_raman_spectra/`.
- Raw Raman download support has been added in `../../../src/download_mlrod_raw_raman.py`.
- Full raw Raman download is not yet complete because direct ODR file downloads are slow from the current network. The current raw-file manifest is `../raw_raman_download_manifest.csv`.

## Dataset Scale

The ODR metadata catalogue lists 19 sample groups and 128,841 spectra:

- 15 training sample groups, 89,121 spectra.
- 4 dusty/undusted rock test sample groups, 39,720 spectra.

## Overlap With the MST Raman Project

| MLROD sample group | MLROD mineral(s) | MST superclass mapping | Recommended use |
|---|---:|---|---|
| Feldspar - Albite | Albite | Tectosilicate | Candidate training or external validation |
| Feldspar - Anorthite | Anorthite | Tectosilicate | Candidate training or external validation |
| Feldspar - Microcline | Microcline | Tectosilicate | Candidate training or external validation |
| Pyroxene - Augite | Augite | Pyroxene | Candidate training or external validation |
| Pyroxene - Enstatite | Enstatite | Pyroxene | Candidate training or external validation |
| Olivine - Forsterite | Forsterite | Olivine | Candidate training or external validation |
| Carbonate - Calcite | Calcite | Carbonate | Candidate training or external validation |
| Sulfate - Gypsum | Gypsum | Sulfate | Candidate training or external validation |
| Mica - Biotite | Biotite | Phyllosilicate | Candidate training or external validation |
| Mica - Muscovite | Muscovite | Phyllosilicate | Candidate training or external validation |
| Quartz | Quartz | Silica Phase | Candidate training or external validation |
| Amphibole - Hornblende | Hornblende | Other Silicate | Candidate training or external validation |
| 50% Forsterite / 50% Albite Mix | Forsterite + Albite | mixture_or_rock | External robustness only |
| 50% Forsterite / 50% Augite Mix | Forsterite + Augite | mixture_or_rock | External robustness only |
| 50% Quartz / 50% Albite Mix | Quartz + Albite | mixture_or_rock | External robustness only |
| Gabbro/Granite rock slabs with 0% or 50% dust | Multi-phase rock spectra | rock_test_set | External robustness only |

## Recommended Manuscript Use

The single-mineral MLROD spectra overlap strongly with the manuscript classes and can be used as an external Raman benchmark after source tagging and QC. The rock slabs and binary mixtures should not be merged into the closed-set single-mineral training set unless the manuscript is changed to a multi-label or mixture-aware formulation; they are better suited for robustness testing, dust/noise sensitivity, and rejection-threshold analysis.
