# SHERLOC Region Point-Spectrum Dataset

This folder contains metadata generated from the local SHERLOC Mars 2020 region
files for Dourbes, Garde, Guillaumes, and Quartier. The build script is
`build_sherloc_region_dataset.py` in the project root.

## Source Inputs

- Region folders: `D:\dyt\raman\metadata\mars\mars\区域\{dourbes,garde,guillaumes,quartier}`
- Standard-label workbooks:
  - `标准数据D.xlsx`
  - `标准数据G.xlsx`
  - `标准数据GU.xlsx`
  - `标准数据Q.xlsx`
- SHERLOC all-points Raman CSV files beginning with `ss__`.
- Literature source: Corpolongo et al. (2023), *SHERLOC Raman mineral class detections of the Mars 2020 crater floor campaign*, JGR: Planets.

## Generated Outputs

- `sherloc_region_detail_to_ss_mapping.csv`:
  Auditable mapping between each standard-label workbook sheet and the matched
  `ss__*_R1_AllPoints_Raman.csv` file.

- `metadata_sherloc_region_points_only.csv`:
  Metadata for the newly extracted SHERLOC point spectra only.

- `metadata_parent_945_plus_sherloc_regions.csv`:
  The original 945 parent spectra metadata table with the newly extracted
  SHERLOC point-label spectra appended.

- `metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv`:
  The original 945 parent spectra plus only those SHERLOC point-label rows that
  can be mapped unambiguously to a manuscript Table 1 mineral superclass.

- `sherloc_region_point_extraction_manifest.csv`:
  One row per exported point-label spectrum, including status, region, sheet,
  point, raw label, harmonized label, and source Raman CSV.

- `sherloc_region_dataset_summary.csv`:
  Counts by region, harmonized mineral class, and label status.

- `sherloc_region_table1_training_summary.csv`:
  Training-usable SHERLOC counts summarized by manuscript Table 1 superclass
  and region.

- `sherloc_region_mapping_basis.md`:
  Plain-language documentation of the sheet-to-scan-to-`ss__` mapping evidence.

- `../sherloc_corpolongo_region_spectra/*.csv`:
  One exported spectrum per point-label row. Each file contains two columns:
  `raman_shift_cm-1` and `intensity`.

## Label Handling

Single-label point assignments are exported directly. If a point has two or
three mineral assignments in the standard workbook, the same point spectrum is
represented by separate metadata rows, one for each assignment, as requested.

Labels that belong to the manuscript Table 1 taxonomy are marked as
`supported_table1_closed_set`. Broad SHERLOC labels such as `silicate` are
mapped to `Other Silicate` and explicitly marked as broad labels. `Na
perchlorate` is mapped to the Table 1 `Perchlorate` superclass. Ambiguous labels
such as `perchlorate/phospate` and `perchlorate/chlorate` are retained as
traceable labels but are marked `sherloc_training_label_usable=False` so they do
not silently enter closed-set model training.

## Reproducibility Notes

The sheet-to-Raman-file mapping is order matched using the scan list reported by
Corpolongo et al. (2023) and the local SHERLOC filenames. Raman CSV files are
sorted by sol, SRLC sequence, and w108/w208 subscan order; workbook sheets follow
the corresponding HDR/detail scan order.
