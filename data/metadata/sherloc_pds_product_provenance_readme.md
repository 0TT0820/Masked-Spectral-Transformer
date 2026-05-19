# SHERLOC PDS Product Provenance

`sherloc_pds_product_provenance.csv` links each local SHERLOC-derived point
spectrum group to the corresponding NASA PDS Mars 2020 SHERLOC processed
spectroscopy product.

The table is built from the official SHERLOC filename convention documented in
the Mars 2020 SHERLOC RDR SIS. For `RRS` products, the local spectra are traced
to processed target spectra in the PDS SHERLOC `data_processed` collection. The
PDS4 XML label URL in each row should be treated as the authoritative product
metadata record.

Important columns:

- `product_stem`: PDS product identifier without file extension.
- `pds_product_lid`: PDS logical identifier for the processed product.
- `pds_csv_url` / `pds_label_url`: direct PDS CSV and XML label URLs.
- `sol`, `sclk`, `sub_sclk`, `site`, `drive`, `sequence`: fields parsed from
  the official SHERLOC filename.
- `proc_flag_1` through `proc_flag_4`: processing flags from the filename.
  For example, `w108cgn` means wavelength correction (`w`), experiment ID `1`,
  ACI image number `08`, cosmic-ray correction (`c`), gain correction (`g`),
  and laser normalization (`n`).
- `pds_raw_product_lids` and `pds_intermediate_product_lids`: upstream PDS
  products extracted from XML labels when the script is run with
  `--fetch-labels`.

For one-row-per-spectrum traceability, use
`sherloc_spectrum_to_pds_crosswalk.csv`. That table links each local spectrum
row to the PDS product identifier, CSV URL, XML label URL, mineral label, point
name, target, and scan metadata.

Regenerate with:

```bash
python src/build_sherloc_pds_provenance.py --fetch-labels
```

Use `--no-fetch-labels` for an offline filename-only table.
