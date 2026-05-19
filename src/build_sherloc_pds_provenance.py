"""Build product-level PDS provenance for SHERLOC spectra.

The local SHERLOC point spectra are derived from PDS SHERLOC processed
spectroscopy RDR products. This script parses the official Mars 2020 SHERLOC
filename fields encoded in the local `ss__...` names and writes a compact
product-level provenance table. Network access is optional: when enabled, the
script also reads the PDS4 XML labels to capture observation times and
raw/intermediate product references.
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDS_BASE_URL = (
    "https://pds-geosciences.wustl.edu/m2020/"
    "urn-nasa-pds-mars2020_sherloc/data_processed"
)
PDS_BUNDLE_LID = "urn:nasa:pds:mars2020_sherloc"
PDS_PROCESSED_COLLECTION_LID = "urn:nasa:pds:mars2020_sherloc:data_processed"
PDS_BUNDLE_DOI = "10.17189/1522643"


PRODUCT_RE = re.compile(
    r"(?P<product_stem>ss__"
    r"(?P<sol>\d{4})_"
    r"(?P<sclk>\d{10})_"
    r"(?P<sub_sclk>\d{3})"
    r"(?P<product_id>[a-z0-9]{3})__"
    r"(?P<site>[0-9a-z]{3})"
    r"(?P<drive>[0-9a-z]{4})"
    r"(?P<sequence>srlc\d{5})"
    r"(?P<proc_flag_1>[a-z_])"
    r"(?P<experiment_id>[0-9a-z_])"
    r"(?P<aci_image_number>[0-9a-z_]{2})"
    r"(?P<proc_flag_2>[a-z_])"
    r"(?P<proc_flag_3>[a-z_])"
    r"(?P<proc_flag_4>[a-z_])"
    r"(?P<producer>[a-z_])"
    r"(?P<version>[0-9a-z_]{2}))",
    re.IGNORECASE,
)


FLAG_1 = {
    "w": "wavelength correction applied",
    "b": "on-board Process_Data product is background-subtracted",
    "_": "no wavelength correction flag",
}
FLAG_2 = {
    "c": "cosmic-ray correction applied",
    "z": "cosmic-ray correction skipped by configuration",
    "_": "no cosmic-ray correction flag",
}
FLAG_3 = {
    "g": "gain correction applied",
    "p": "on-board Process_Data algorithm applied",
    "_": "no gain correction flag",
}
FLAG_4 = {
    "n": "laser normalization applied",
    "z": "laser normalization skipped by configuration",
    "_": "no laser normalization flag",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_product_stem(text: str) -> dict[str, str] | None:
    match = PRODUCT_RE.search(text)
    if not match:
        return None
    fields = {key: value.lower() for key, value in match.groupdict().items()}
    fields["sol_int"] = str(int(fields["sol"]))
    fields["sequence_numeric"] = fields["sequence"].replace("srlc", "")
    fields["pds_product_lid"] = (
        f"{PDS_PROCESSED_COLLECTION_LID}:{fields['product_stem']}"
    )
    sol_dir = f"sol_{int(fields['sol']):05d}"
    fields["pds_csv_url"] = (
        f"{PDS_BASE_URL}/{sol_dir}/{fields['product_stem']}.csv"
    )
    fields["pds_label_url"] = (
        f"{PDS_BASE_URL}/{sol_dir}/{fields['product_stem']}.xml"
    )
    fields["pds_sol_directory"] = sol_dir
    fields["product_type_description"] = product_type_description(
        fields["product_id"]
    )
    fields["processing_flags_description"] = "; ".join(
        [
            FLAG_1.get(fields["proc_flag_1"], "unknown first processing flag"),
            FLAG_2.get(fields["proc_flag_2"], "unknown second processing flag"),
            FLAG_3.get(fields["proc_flag_3"], "unknown third processing flag"),
            FLAG_4.get(fields["proc_flag_4"], "unknown fourth processing flag"),
        ]
    )
    fields["release_number"] = release_for_sol(int(fields["sol"]))
    return fields


def product_type_description(product_id: str) -> str:
    if product_id.lower() == "rrs":
        return (
            "RRS: dark-subtracted spectral data of a non-internal target; "
            "additional processing is indicated by the filename flags."
        )
    return f"{product_id.upper()}: see SHERLOC RDR SIS product-type table."


def release_for_sol(sol: int) -> str:
    if 90 <= sol <= 179:
        return "Mars 2020 Release 2 (sols 90-179; public release 2021-11-22)"
    if 180 <= sol <= 299:
        return "Mars 2020 Release 3 (sols 180-299; public release 2022-03-22)"
    if 300 <= sol <= 419:
        return "Mars 2020 Release 4 (sols 300-419; public release 2022-07-22)"
    return "See Mars 2020 PDS release schedule"


def fetch_label_metadata(label_url: str, timeout: int = 30) -> dict[str, str]:
    out = {
        "pds_label_fetch_status": "not_requested",
        "pds_start_date_time": "",
        "pds_stop_date_time": "",
        "pds_earth_received_start_date_time": "",
        "pds_earth_received_stop_date_time": "",
        "pds_raw_product_lids": "",
        "pds_intermediate_product_lids": "",
        "pds_document_lids": "",
    }
    try:
        with urllib.request.urlopen(label_url, timeout=timeout) as response:
            xml_text = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        out["pds_label_fetch_status"] = f"failed: {exc}"
        return out

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        out["pds_label_fetch_status"] = f"parse_failed: {exc}"
        return out

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    refs: dict[str, list[str]] = defaultdict(list)
    for elem in root.iter():
        name = local_name(elem.tag)
        text = (elem.text or "").strip()
        if name == "start_date_time":
            out["pds_start_date_time"] = text
        elif name == "stop_date_time":
            out["pds_stop_date_time"] = text
        elif name == "earth_received_start_date_time":
            out["pds_earth_received_start_date_time"] = text
        elif name == "earth_received_stop_date_time":
            out["pds_earth_received_stop_date_time"] = text

    for ref in root.iter():
        if local_name(ref.tag) != "Internal_Reference":
            continue
        lid = ""
        ref_type = ""
        for child in ref:
            cname = local_name(child.tag)
            ctext = (child.text or "").strip()
            if cname == "lid_reference":
                lid = ctext
            elif cname == "reference_type":
                ref_type = ctext
        if not lid:
            continue
        if ref_type == "data_to_raw_product":
            refs["raw"].append(lid)
        elif ref_type == "data_to_calibrated_product":
            refs["intermediate"].append(lid)
        elif ref_type == "data_to_document":
            refs["document"].append(lid)

    out["pds_raw_product_lids"] = "|".join(refs["raw"])
    out["pds_intermediate_product_lids"] = "|".join(refs["intermediate"])
    out["pds_document_lids"] = "|".join(refs["document"])
    out["pds_label_fetch_status"] = "ok"
    return out


def collect_parent_products(parent_path: Path) -> dict[str, dict[str, object]]:
    products: dict[str, dict[str, object]] = {}
    rows = read_csv(parent_path)
    for row in rows:
        if row.get("source_type") != "SHERLOC in-situ spectra":
            continue
        text = row.get("file_name_clean") or row.get("file_name") or ""
        parsed = parse_product_stem(text)
        if parsed is None:
            continue
        product = products.setdefault(parsed["product_stem"], init_product(parsed))
        product["dataset_memberships"].add("parent_945_sherloc_rows")
        product["point_spectrum_count"] += 1
        product["mineral_labels"].add(row.get("mineral_species") or row.get("major_category") or "")
        product["major_categories"].add(row.get("major_category") or "")
        product["source_spectrum_ids"].add(row.get("spectrum_id") or "")
        product["local_source_files"].add(row.get("file_path") or row.get("file_name_clean") or "")
        infer_original_target(product, parsed)
    return products


def collect_region_products(region_mapping_path: Path) -> dict[str, dict[str, object]]:
    products: dict[str, dict[str, object]] = {}
    rows = read_csv(region_mapping_path)
    for row in rows:
        text = row.get("ss_raman_file") or ""
        parsed = parse_product_stem(text)
        if parsed is None:
            continue
        product = products.setdefault(parsed["product_stem"], init_product(parsed))
        product["dataset_memberships"].add("sherloc_region_point_dataset")
        product["regions"].add(row.get("region") or "")
        product["targets"].add(row.get("target") or "")
        product["scan_names"].add(row.get("scan_name") or "")
        product["sheet_names"].add(row.get("sheet_name") or "")
        product["local_source_files"].add(Path(text).name)
    return products


def collect_region_point_products(region_points_path: Path) -> dict[str, dict[str, object]]:
    products: dict[str, dict[str, object]] = {}
    rows = read_csv(region_points_path)
    for row in rows:
        text = row.get("sherloc_source_raman_csv") or row.get("source_id") or ""
        parsed = parse_product_stem(text)
        if parsed is None:
            continue
        product = products.setdefault(parsed["product_stem"], init_product(parsed))
        product["dataset_memberships"].add("sherloc_region_point_dataset")
        product["point_spectrum_count"] += 1
        product["regions"].add(row.get("sherloc_region") or "")
        product["targets"].add(row.get("sherloc_target") or "")
        product["scan_names"].add(row.get("sherloc_scan_name") or "")
        product["sheet_names"].add(row.get("sherloc_sheet_name") or "")
        product["mineral_labels"].add(row.get("mineral_species") or row.get("major_category") or "")
        product["major_categories"].add(row.get("major_category") or "")
        product["source_spectrum_ids"].add(row.get("spectrum_id") or "")
        product["local_source_files"].add(row.get("file_path") or row.get("file_name_clean") or "")
    return products


def init_product(parsed: dict[str, str]) -> dict[str, object]:
    item: dict[str, object] = dict(parsed)
    item.update(
        {
            "dataset_memberships": set(),
            "regions": set(),
            "targets": set(),
            "scan_names": set(),
            "sheet_names": set(),
            "mineral_labels": set(),
            "major_categories": set(),
            "source_spectrum_ids": set(),
            "local_source_files": set(),
            "point_spectrum_count": 0,
        }
    )
    return item


def infer_original_target(product: dict[str, object], parsed: dict[str, str]) -> None:
    """Add target hints for original parent rows when literature mapping exists."""
    sol = int(parsed["sol"])
    sequence = parsed["sequence_numeric"]
    target = ""
    scan = ""
    if sol == 186:
        target = "Bellegarde"
        scan = "0186_Bellegarde; exact scan name to verify against mission label set"
    elif sol == 304 and sequence == "11374":
        target = "Quartier"
        scan = "0304_Quartier Detail_500_3, experiment 1 / ACI image 08"
    if target:
        product["targets"].add(target)
        product["scan_names"].add(scan)
        product["regions"].add(target.lower())


def merge_products(*tables: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for table in tables:
        for stem, product in table.items():
            if stem not in merged:
                merged[stem] = product
                continue
            dest = merged[stem]
            for key, value in product.items():
                if isinstance(value, set):
                    dest.setdefault(key, set()).update(value)
                elif isinstance(value, int):
                    dest[key] = int(dest.get(key, 0)) + value
                elif not dest.get(key):
                    dest[key] = value
    return merged


def format_set(value: object) -> str:
    if not isinstance(value, set):
        return str(value or "")
    return "|".join(sorted(v for v in value if v))


def write_outputs(products: dict[str, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "product_stem",
        "dataset_memberships",
        "point_spectrum_count",
        "regions",
        "targets",
        "scan_names",
        "sheet_names",
        "mineral_labels",
        "major_categories",
        "pds_product_lid",
        "pds_bundle_lid",
        "pds_processed_collection_lid",
        "pds_bundle_doi",
        "pds_sol_directory",
        "pds_csv_url",
        "pds_label_url",
        "release_number",
        "sol",
        "sclk",
        "sub_sclk",
        "product_id",
        "product_type_description",
        "site",
        "drive",
        "sequence",
        "sequence_numeric",
        "proc_flag_1",
        "experiment_id",
        "aci_image_number",
        "proc_flag_2",
        "proc_flag_3",
        "proc_flag_4",
        "processing_flags_description",
        "producer",
        "version",
        "pds_label_fetch_status",
        "pds_start_date_time",
        "pds_stop_date_time",
        "pds_earth_received_start_date_time",
        "pds_earth_received_stop_date_time",
        "pds_raw_product_lids",
        "pds_intermediate_product_lids",
        "pds_document_lids",
        "provenance_note",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for product in sorted(
            products.values(), key=lambda item: (item["sol"], item["sclk"], item["sub_sclk"])
        ):
            row = {key: format_set(product.get(key, "")) for key in fieldnames}
            row["pds_bundle_lid"] = PDS_BUNDLE_LID
            row["pds_processed_collection_lid"] = PDS_PROCESSED_COLLECTION_LID
            row["pds_bundle_doi"] = PDS_BUNDLE_DOI
            row["provenance_note"] = (
                "Local point spectra are derived from this PDS SHERLOC processed "
                "spectroscopy RRS product. The PDS4 XML label is the authoritative "
                "product-level metadata source."
            )
            writer.writerow(row)


def build_crosswalk(
    parent_path: Path,
    region_points_path: Path,
    products: dict[str, dict[str, object]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def product_fields(text: str) -> dict[str, str] | None:
        parsed = parse_product_stem(text)
        if parsed is None:
            return None
        product = products.get(parsed["product_stem"], {})
        return {
            "product_stem": parsed["product_stem"],
            "pds_product_lid": parsed["pds_product_lid"],
            "pds_csv_url": parsed["pds_csv_url"],
            "pds_label_url": parsed["pds_label_url"],
            "pds_raw_product_lids": str(product.get("pds_raw_product_lids", "")),
            "pds_intermediate_product_lids": str(
                product.get("pds_intermediate_product_lids", "")
            ),
            "pds_start_date_time": str(product.get("pds_start_date_time", "")),
            "pds_stop_date_time": str(product.get("pds_stop_date_time", "")),
            "sol": parsed["sol"],
            "sclk": parsed["sclk"],
            "sub_sclk": parsed["sub_sclk"],
            "site": parsed["site"],
            "drive": parsed["drive"],
            "sequence": parsed["sequence"],
            "experiment_id": parsed["experiment_id"],
            "aci_image_number": parsed["aci_image_number"],
            "processing_flags_description": parsed["processing_flags_description"],
        }

    for row in read_csv(parent_path):
        if row.get("source_type") != "SHERLOC in-situ spectra":
            continue
        fields = product_fields(row.get("file_name_clean") or row.get("file_name") or "")
        if fields is None:
            continue
        target = "Bellegarde" if fields["sol"] == "0186" else ("Quartier" if fields["sol"] == "0304" else "")
        out = {
            "dataset_membership": "parent_945_sherloc_rows",
            "spectrum_id": row.get("spectrum_id", ""),
            "local_spectrum_path": row.get("file_path", ""),
            "local_source_file": row.get("file_name_clean", ""),
            "region": target.lower(),
            "target": target,
            "scan_name": "",
            "sheet_name": "",
            "point_name": row.get("file_name_clean", "").rsplit("-", 1)[-1],
            "mineral_label": row.get("mineral_species") or row.get("major_category", ""),
            "major_category": row.get("major_category", ""),
            "label_basis": row.get("label_basis", ""),
            "reference": row.get("reference", ""),
        }
        out.update(fields)
        rows.append(out)

    for row in read_csv(region_points_path):
        fields = product_fields(row.get("sherloc_source_raman_csv") or row.get("source_id") or "")
        if fields is None:
            continue
        out = {
            "dataset_membership": "sherloc_region_point_dataset",
            "spectrum_id": row.get("spectrum_id", ""),
            "local_spectrum_path": row.get("file_path", ""),
            "local_source_file": row.get("sherloc_source_raman_csv", ""),
            "region": row.get("sherloc_region", ""),
            "target": row.get("sherloc_target", ""),
            "scan_name": row.get("sherloc_scan_name", ""),
            "sheet_name": row.get("sherloc_sheet_name", ""),
            "point_name": row.get("sherloc_point_name", ""),
            "mineral_label": row.get("mineral_species") or row.get("major_category", ""),
            "major_category": row.get("major_category", ""),
            "label_basis": row.get("label_basis", ""),
            "reference": row.get("reference", ""),
        }
        out.update(fields)
        rows.append(out)
    return rows


def write_crosswalk(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_path: Path, readme_path: Path) -> None:
    readme_path.write_text(
        """# SHERLOC PDS Product Provenance

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
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent-metadata",
        type=Path,
        default=ROOT / "data" / "metadata" / "metadata_parent_945.csv",
    )
    parser.add_argument(
        "--region-mapping",
        type=Path,
        default=ROOT / "data" / "metadata" / "sherloc_region_detail_to_ss_mapping.csv",
    )
    parser.add_argument(
        "--region-points",
        type=Path,
        default=ROOT / "data" / "metadata" / "metadata_sherloc_region_points_only.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "metadata" / "sherloc_pds_product_provenance.csv",
    )
    parser.add_argument(
        "--crosswalk-output",
        type=Path,
        default=ROOT / "data" / "metadata" / "sherloc_spectrum_to_pds_crosswalk.csv",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=ROOT / "data" / "metadata" / "sherloc_pds_product_provenance_readme.md",
    )
    parser.add_argument("--fetch-labels", action="store_true")
    parser.add_argument("--no-fetch-labels", action="store_true")
    args = parser.parse_args()

    products = merge_products(
        collect_parent_products(args.parent_metadata),
        collect_region_products(args.region_mapping),
        collect_region_point_products(args.region_points),
    )
    if args.fetch_labels and not args.no_fetch_labels:
        for product in products.values():
            product.update(fetch_label_metadata(str(product["pds_label_url"])))
    else:
        for product in products.values():
            product.update(fetch_label_metadata("", timeout=1))

    write_outputs(products, args.output)
    write_crosswalk(
        build_crosswalk(args.parent_metadata, args.region_points, products),
        args.crosswalk_output,
    )
    write_readme(args.output, args.readme)
    print(f"Wrote {len(products)} SHERLOC PDS product rows to {args.output}")


if __name__ == "__main__":
    main()
