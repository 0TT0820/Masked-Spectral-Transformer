"""Build a PDS-traceable SHERLOC SaU 008 calibration-target dataset.

The SHERLOC external calibration target palette includes Mars meteorite
SaU 008. In SHERLOC product identifiers, external calibration products use
`SRLC15*`; the calibration-target material index 3 is SaU 008, so this script
collects RRS products whose sequence id starts with `SRLC1503`.

The output is intentionally conservative. The PDS RRS files identify the
calibration target as a bulk meteorite target, but they do not provide
point-level mineral labels. Therefore, the extracted spectra are marked as
usable for SHERLOC-domain adaptation or external Mars-domain checking, not as
closed-set supervised mineral labels unless manually annotated later.
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDS_BASE_URL = (
    "https://pds-geosciences.wustl.edu/m2020/"
    "urn-nasa-pds-mars2020_sherloc/data_processed"
)
PDS_COLLECTION_LID = "urn:nasa:pds:mars2020_sherloc:data_processed"
PDS_DOI = "10.17189/1522643"
LASER_WAVELENGTH_NM = 248.6

EXTERNAL_DIR = ROOT / "data" / "external_sherloc_sau008_pds"
PRODUCT_DIR = EXTERNAL_DIR / "products"
METADATA_DIR = ROOT / "data" / "metadata"
OVERVIEW_DIR = ROOT / "data" / "overview" / "sherloc_sau008_calibration"
SPECTRA_DIR = ROOT / "data" / "sherloc_sau008_calibration" / "mean_spectra"
DOCS_DIR = ROOT / "docs"

INVENTORY_PATH = EXTERNAL_DIR / "collection_data_processed_inventory.csv"

PRODUCT_RE = re.compile(
    r"(?P<stem>ss__(?P<sol>\d{4})_\d{10}_\d{3}rrs__"
    r"[0-9a-z]{7}(?P<sequence>srlc1503[01])[0-9a-z_]+)",
    re.IGNORECASE,
)


def ensure_dirs() -> None:
    for path in (EXTERNAL_DIR, PRODUCT_DIR, METADATA_DIR, OVERVIEW_DIR, SPECTRA_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_if_missing(url: str, path: Path, timeout: int = 120) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=timeout) as response:
        path.write_bytes(response.read())


def read_inventory_products() -> list[dict[str, str]]:
    products: list[dict[str, str]] = []
    with INVENTORY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            lid = row[1].strip()
            match = PRODUCT_RE.search(lid)
            if not match:
                continue
            stem = match.group("stem").lower()
            sol = int(match.group("sol"))
            sequence = match.group("sequence").lower()
            sol_dir = f"sol_{sol:05d}"
            products.append(
                {
                    "product_stem": stem,
                    "sol": str(sol),
                    "sol_dir": sol_dir,
                    "sequence_id": sequence,
                    "pds_product_lid": f"{PDS_COLLECTION_LID}:{stem}",
                    "pds_csv_url": f"{PDS_BASE_URL}/{sol_dir}/{stem}.csv",
                    "pds_label_url": f"{PDS_BASE_URL}/{sol_dir}/{stem}.xml",
                    "local_csv": str(PRODUCT_DIR / f"{stem}.csv"),
                    "local_xml": str(PRODUCT_DIR / f"{stem}.xml"),
                }
            )
    products.sort(key=lambda item: (int(item["sol"]), item["product_stem"]))
    return products


def parse_start_time(xml_path: Path) -> str:
    if not xml_path.exists() or xml_path.stat().st_size == 0:
        return ""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return ""
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag == "start_date_time" and elem.text:
            return elem.text.strip()
    return ""


def parse_float_row(line: str) -> list[float]:
    return [float(value) for value in line.strip().split(",") if value.strip()]


def wavelength_to_raman_shift(wavelength_nm: float) -> float:
    return (1.0 / LASER_WAVELENGTH_NM - 1.0 / wavelength_nm) * 1.0e7


def iter_rrs_regions(csv_path: Path):
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError(f"RRS file is too short: {csv_path}")

    wavelength_idx = None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("WAVELENGTH"):
            wavelength_idx = idx + 2
            break
    if wavelength_idx is None:
        raise ValueError(f"Could not find wavelength row in {csv_path}")
    wavelengths = parse_float_row(lines[wavelength_idx])
    raman_shifts = [wavelength_to_raman_shift(value) for value in wavelengths]

    region_indices = [
        idx for idx, line in enumerate(lines)
        if (
            "SPECTRA:_REGION_" in line.upper()
            or line.upper().startswith("PROCESS_DATA_SPECTRUM_REGION_")
        )
    ]
    for pos, marker_idx in enumerate(region_indices):
        marker = lines[marker_idx].strip()
        region_name = marker.split(":")[-1].strip() if ":" in marker else marker
        region_match = re.search(r"REGION[_ ]*(\d+)", region_name, re.IGNORECASE)
        if region_match:
            region_name = f"REGION_{region_match.group(1)}"
        data_start = marker_idx + 2
        data_end = region_indices[pos + 1] if pos + 1 < len(region_indices) else len(lines)
        spectra: list[list[float]] = []
        for line in lines[data_start:data_end]:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values = parse_float_row(stripped)
            except ValueError:
                continue
            if len(values) == len(wavelengths):
                spectra.append(values)
        yield region_name, wavelengths, raman_shifts, spectra


def write_mean_spectrum(path: Path, raman_shifts: list[float], wavelengths: list[float], mean: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["raman_shift_cm-1", "wavelength_nm", "intensity"])
        for shift, wavelength, intensity in zip(raman_shifts, wavelengths, mean):
            writer.writerow([f"{shift:.6f}", f"{wavelength:.6f}", f"{intensity:.6f}"])


def mean_spectrum(spectra: list[list[float]]) -> list[float]:
    if not spectra:
        return []
    sums = [0.0] * len(spectra[0])
    for row in spectra:
        for idx, value in enumerate(row):
            sums[idx] += value
    return [value / len(spectra) for value in sums]


def build_dataset(download: bool) -> None:
    ensure_dirs()
    products = read_inventory_products()

    mean_rows: list[dict[str, str]] = []
    point_rows: list[dict[str, str]] = []
    product_rows: list[dict[str, str]] = []

    for product in products:
        csv_path = Path(product["local_csv"])
        xml_path = Path(product["local_xml"])
        if download:
            download_if_missing(product["pds_csv_url"], csv_path)
            download_if_missing(product["pds_label_url"], xml_path)
        if not csv_path.exists():
            continue

        start_time = parse_start_time(xml_path)
        region_count = 0
        point_count = 0
        for region_name, wavelengths, raman_shifts, spectra in iter_rrs_regions(csv_path):
            region_count += 1
            point_count += len(spectra)
            clean_region = region_name.lower().replace(" ", "_")
            spectrum_id = f"SAU008_{product['product_stem']}_{clean_region}_mean".upper()
            output_name = f"{spectrum_id}.csv"
            output_path = SPECTRA_DIR / output_name
            mean = mean_spectrum(spectra)
            if mean:
                write_mean_spectrum(output_path, raman_shifts, wavelengths, mean)

            mean_rows.append(
                {
                    "spectrum_id": spectrum_id,
                    "spectrum_file": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_type": "SHERLOC calibration target Mars meteorite SaU 008",
                    "instrument": "SHERLOC, Perseverance rover",
                    "pds_product_lid": product["pds_product_lid"],
                    "pds_csv_url": product["pds_csv_url"],
                    "pds_label_url": product["pds_label_url"],
                    "pds_bundle_doi": PDS_DOI,
                    "sol": product["sol"],
                    "start_time_utc": start_time,
                    "sequence_id": product["sequence_id"],
                    "calibration_target_number": "3",
                    "calibration_target_name": "Mars Meteorite SaU 008",
                    "region": region_name,
                    "n_point_spectra_averaged": str(len(spectra)),
                    "n_channels": str(len(wavelengths)),
                    "laser_wavelength_nm": f"{LASER_WAVELENGTH_NM:.1f}",
                    "major_category": "",
                    "mineral_label": "",
                    "label_status": "bulk_target_only_no_point_level_mineral_label",
                    "supervised_training_usable": "False",
                    "domain_adaptation_usable": "True",
                    "recommended_use": "SHERLOC-domain adaptation, calibration-domain checking, or manual mineral-label review",
                    "provenance_note": (
                        "External calibration products use SRLC15*; SHERLOC calibration target material "
                        "number 3 is Mars Meteorite SaU 008 in the PDS user guide and RDR SIS."
                    ),
                }
            )

            for point_idx in range(1, len(spectra) + 1):
                point_rows.append(
                    {
                        "point_spectrum_id": f"{spectrum_id}_POINT_{point_idx:03d}",
                        "source_rrs_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                        "pds_product_lid": product["pds_product_lid"],
                        "pds_csv_url": product["pds_csv_url"],
                        "sol": product["sol"],
                        "sequence_id": product["sequence_id"],
                        "calibration_target_name": "Mars Meteorite SaU 008",
                        "region": region_name,
                        "point_index_within_region": str(point_idx),
                        "n_channels": str(len(wavelengths)),
                        "label_status": "bulk_target_only_no_point_level_mineral_label",
                        "supervised_training_usable": "False",
                        "domain_adaptation_usable": "True",
                    }
                )

        product_rows.append(
            {
                **product,
                "local_csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "local_xml": str(xml_path.relative_to(ROOT)).replace("\\", "/") if xml_path.exists() else "",
                "start_time_utc": start_time,
                "calibration_target_number": "3",
                "calibration_target_name": "Mars Meteorite SaU 008",
                "region_count": str(region_count),
                "point_spectrum_count": str(point_count),
            }
        )

    write_dicts(METADATA_DIR / "metadata_sherloc_sau008_calibration_mean_spectra.csv", mean_rows)
    write_dicts(METADATA_DIR / "metadata_sherloc_sau008_calibration_point_index.csv", point_rows)
    write_dicts(OVERVIEW_DIR / "sherloc_sau008_pds_products.csv", product_rows)
    write_summary(OVERVIEW_DIR / "sherloc_sau008_summary.csv", product_rows, mean_rows, point_rows)
    write_doc(DOCS_DIR / "sherloc_sau008_calibration_data.md", product_rows, mean_rows, point_rows)
    write_optional_combined_manifest(mean_rows)


def write_dicts(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_optional_combined_manifest(mean_rows: list[dict[str, str]]) -> None:
    base_path = METADATA_DIR / "metadata_training_ready_plus_martian_meteorite_mendeley.csv"
    output_path = METADATA_DIR / "metadata_training_ready_plus_sau008_domain_adaptation.csv"
    if not base_path.exists():
        return

    with base_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        base_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    extra_fields = [
        "sau008_sequence_id",
        "sau008_calibration_target_number",
        "sau008_calibration_target_name",
        "sau008_region",
        "sau008_n_point_spectra_averaged",
        "sau008_pds_product_lid",
        "sau008_pds_csv_url",
        "sau008_pds_label_url",
        "domain_adaptation_usable",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    combined_rows = [{field: row.get(field, "") for field in fieldnames} for row in base_rows]
    for row in mean_rows:
        combined = {field: "" for field in fieldnames}
        combined.update(
            {
                "file_name": f"{row['spectrum_id']}.csv",
                "group_label": "",
                "subtype_label": "",
                "major_category": "",
                "file_path": row["spectrum_file"],
                "file_exists": "True",
                "spectrum_id": row["spectrum_id"],
                "mineral_species": "",
                "source_id": row["pds_product_lid"],
                "source_type": row["source_type"],
                "spectrum_type": "SHERLOC DUV Raman calibration-target mean spectrum",
                "excitation_nm": row["laser_wavelength_nm"],
                "instrument": row["instrument"],
                "data_level": "PDS RRS processed region mean",
                "sample_provenance": "Mars meteorite SaU 008 mounted on the SHERLOC external calibration target",
                "measurement_conditions": (
                    f"Mars 2020 SHERLOC calibration observation on sol {row['sol']}; "
                    f"PDS RRS product {row['pds_product_lid']}; {row['region']}; "
                    f"mean of {row['n_point_spectra_averaged']} point spectra."
                ),
                "label_basis": "Bulk calibration-target identity only; no point-level mineral label in PDS RRS product.",
                "reference": f"Mars 2020 SHERLOC PDS processed collection, DOI {PDS_DOI}; SHERLOC User Guide; SHERLOC RDR SIS.",
                "source_note": row["provenance_note"],
                "n_original_points": row["n_channels"],
                "preprocessing_planned": "Use only in domain-adaptation or manual-label review workflow unless mineral labels are added.",
                "augmentation_used": "False",
                "qc_status": "traceable_bulk_target_no_supervised_label",
                "qc_reason": row["label_status"],
                "recommended_action": row["recommended_use"],
                "split_main": "domain_adaptation_only",
                "split_zero_shot_protocol": "excluded_from_supervised_label_splits",
                "sherloc_training_label_usable": "False",
                "training_label_usable": "False",
                "sau008_sequence_id": row["sequence_id"],
                "sau008_calibration_target_number": row["calibration_target_number"],
                "sau008_calibration_target_name": row["calibration_target_name"],
                "sau008_region": row["region"],
                "sau008_n_point_spectra_averaged": row["n_point_spectra_averaged"],
                "sau008_pds_product_lid": row["pds_product_lid"],
                "sau008_pds_csv_url": row["pds_csv_url"],
                "sau008_pds_label_url": row["pds_label_url"],
                "domain_adaptation_usable": row["domain_adaptation_usable"],
            }
        )
        combined_rows.append(combined)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)


def write_summary(
    path: Path,
    product_rows: list[dict[str, str]],
    mean_rows: list[dict[str, str]],
    point_rows: list[dict[str, str]],
) -> None:
    by_sol: defaultdict[str, int] = defaultdict(int)
    by_sequence: defaultdict[str, int] = defaultdict(int)
    for row in product_rows:
        by_sol[row["sol"]] += 1
        by_sequence[row["sequence_id"]] += 1

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["pds_rrs_products", len(product_rows)])
        writer.writerow(["mean_region_spectra", len(mean_rows)])
        writer.writerow(["point_spectra_indexed", len(point_rows)])
        writer.writerow(["supervised_training_usable", 0])
        writer.writerow(["domain_adaptation_usable", len(mean_rows)])
        for sol, count in sorted(by_sol.items(), key=lambda item: int(item[0])):
            writer.writerow([f"products_on_sol_{sol}", count])
        for sequence, count in sorted(by_sequence.items()):
            writer.writerow([f"products_with_{sequence}", count])


def write_doc(
    path: Path,
    product_rows: list[dict[str, str]],
    mean_rows: list[dict[str, str]],
    point_rows: list[dict[str, str]],
) -> None:
    text = f"""# SHERLOC SaU 008 Calibration Target Data

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

- PDS processed collection DOI: `{PDS_DOI}`
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

- PDS RRS products: {len(product_rows)}
- Mean region spectra extracted: {len(mean_rows)}
- Point spectra indexed in source RRS files: {len(point_rows)}

The point-level intensities remain in the original downloaded PDS RRS CSV files
under `data/external_sherloc_sau008_pds/products/`. The point index table
records the source product, detector region, and row index needed to recover
each spectrum exactly.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing SaU 008 RRS CSV/XML products from PDS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(download=args.download)


if __name__ == "__main__":
    main()
