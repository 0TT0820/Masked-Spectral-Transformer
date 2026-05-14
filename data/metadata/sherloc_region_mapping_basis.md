# SHERLOC Region-to-File Mapping Basis

The mapping in `sherloc_region_detail_to_ss_mapping.csv` was built from three
independent pieces of evidence.

1. Corpolongo et al. (2023) defines the target/scan naming convention: a target
   is the rock or abraded patch name, and a scan name follows the pattern
   `#0001_name_#2_#3`, where `#0001` is the sol, `name` is the target and scan
   type, `#2` is laser pulses per point, and `#3` differentiates repeated scans.

2. The same paper lists the scans for the four targets used here:
   - Dourbes: `0269_Dourbes Detail_500_1`, `0269_Dourbes Detail_500_2`,
     and `0269_Dourbes Detail_500_3`.
   - Garde: `0208_Garde Detail_500_1`, `0208_Garde Detail_500_2`,
     and `0208_Garde Detail_500_3`.
   - Guillaumes: `0162_Guillaumes HDR_250_1`.
   - Quartier: `0293_Quartier HDR_500_1` and
     `0304_Quartier Detail_500_1` through `0304_Quartier Detail_500_4`.

3. The local SHERLOC all-points Raman filenames encode the sol and SRLC sequence.
   Within each target folder, files were sorted by sol, SRLC sequence, and then
   the `w108` / `w208` subscan suffix. The standard-label workbook sheets follow
   the same order, so paired sheets such as `detail_500_1.1` and
   `detail_500_1.2` are mapped to the two `w108` and `w208` files for the same
   scan number.

This mapping is intentionally stored as a CSV table rather than embedded only in
code, so each spectrum exported from a `Point_x` column remains traceable to the
region, workbook, sheet, SHERLOC scan, original all-points Raman CSV, and the
Corpolongo et al. (2023) literature source.
