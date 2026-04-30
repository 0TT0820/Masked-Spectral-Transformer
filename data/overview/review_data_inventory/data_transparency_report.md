# Review Data Transparency Report

Generated from `data\metadata_outputs\metadata_parent_945.csv`.

## Primary Recommendation

Use a clean zero-shot/transfer-separated protocol in the revised manuscript:

1. Earth-domain training, validation, and test use only RRUFF, laboratory DUV analogue spectra, and confirmed meteorite spectra.
2. SHERLOC in-situ spectra are kept out of the primary training set and reported as external Mars-domain validation.
3. If a SHERLOC fine-tuning experiment is retained, it must be reported as a separate transfer-learning experiment using the product-group split in `sherloc_product_group_summary.csv`.
4. Halides are excluded from the review-ready supervised classification table because the reviewer is correct that pure crystalline halite is weak/non-Raman-active under standard Raman conditions. Halide/chloride detection should be discussed as a separate operational/non-detection or uncertainty problem.

## Dataset Counts

- Parent spectra: 945
- RRUFF parent spectra: 791
- Laboratory DUV parent spectra: 119
- Martian meteorite parent spectra: 4
- SHERLOC external spectra: 31
- Review-ready Earth train parent spectra: 630
- Review-ready Earth validation parent spectra: 134
- Review-ready Earth test parent spectra: 133
- Effective train spectra after reproducible class balancing to 200 per class: 2600

## Files

- `spectrum_level_provenance_review.csv`: one row per parent spectrum with source, role, QC, hash, and reference fields.
- `dataset_flow_by_review_class.csv`: raw parent counts, split counts, external SHERLOC counts, and reproducible augmentation counts by class.
- `source_split_class_matrix.csv`: source x split x class matrix.
- `sherloc_product_group_summary.csv`: SHERLOC product groups and candidate transfer roles.
- `meteorite_spot_inventory.csv`: Tissint/NWA 10153 spectra requiring final spot-level assignment.
- `excluded_halide_inventory.csv`: halide spectra removed from the review-ready classification protocol.

## External Source Notes

- RRUFF: the RRUFF project provides mineral Raman/IR/XRD/chemistry data and asks users to cite the RRUFF Project and external contributors where applicable. See https://www.rruff.net/about/.
- RRUFF Raman limitations: RRUFF Raman data include visible red/green laser excitation; fluorescence behaviour is wavelength dependent, so DUV transfer must be explicitly described. See https://rruff.info/about/about_raman.php.
- SHERLOC archive: NASA PDS provides raw, intermediate, and processed spectroscopy collections for Mars 2020 SHERLOC. See https://pds-geosciences.wustl.edu/missions/mars2020/sherloc.htm.
- Bellegarde/Dourbes context: Razzell Hollis et al. (2022), Icarus, DOI https://doi.org/10.1016/j.icarus.2022.115179.
- Quartier/Bellegarde sulfate context: Wang et al. (2024), Vibrational Spectroscopy, DOI https://doi.org/10.1016/j.vibspec.2024.103745.

## Manual Items Still Required

- Assign each meteorite spectrum (`kuang`, `kuang2`, `py`, `pl-g`) to Tissint or NWA 10153 using the original acquisition log or TIMA/BSE spot map.
- Resolve exact SHERLOC target names for each `ss__...` product group using PDS labels or Analyst's Notebook metadata.
- Complete supplier, lot, and preparation metadata for the laboratory DUV analogue minerals.
