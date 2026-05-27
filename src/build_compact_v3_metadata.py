"""Create a GitHub-friendly compact v3 metadata table.

The full MLROD-integrated metadata table is intentionally verbose and can exceed
GitHub's ordinary 100 MB single-file limit because it preserves many provenance
columns inherited from earlier curation steps. This script writes a compact
table with the columns needed to audit source, label, split, and raw-data access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_METADATA = ROOT / "data" / "metadata" / "metadata_training_database_v3_mlrod_integrated.csv"
OUT_METADATA = ROOT / "data" / "metadata" / "metadata_training_database_v3_compact.csv"

KEEP_COLUMNS = [
    "spectrum_id",
    "file_name",
    "spectrum_storage_format",
    "label_category_final",
    "mineral_species_final",
    "source_type_normalized",
    "training_role",
    "split_v3",
    "split_zero_shot_protocol",
    "source_id",
    "instrument",
    "excitation_nm",
    "spectral_min_cm-1",
    "spectral_max_cm-1",
    "n_original_points",
    "qc_status",
    "qc_reason",
    "recommended_action",
    "supervised_label_usable_v2",
    "duv_library_include",
    "mlrod_odr_datarecord_id",
    "mlrod_file_id",
    "mlrod_sample_id",
    "mlrod_raw_filename",
    "mlrod_row_index",
    "mlrod_relative_container_file",
    "rruff_official_id",
    "rruff_official_name",
    "rruff_url",
    "rruff_status",
    "sherloc_region",
    "sherloc_target",
    "sherloc_scan_name",
    "sherloc_point_name",
]


def main() -> None:
    df = pd.read_csv(FULL_METADATA, low_memory=False)
    root_str = str(ROOT).replace("\\", "/")
    if "file_path" in df.columns:
        df["relative_file_path"] = (
            df["file_path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.replace(root_str + "/", "", regex=False)
        )
    if "mlrod_container_file" in df.columns:
        df["mlrod_relative_container_file"] = (
            df["mlrod_container_file"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.replace(root_str + "/", "", regex=False)
        )
    columns = [column for column in KEEP_COLUMNS if column in df.columns]
    compact = df[columns].copy()
    compact.to_csv(OUT_METADATA, index=False, encoding="utf-8-sig")
    print(f"Wrote compact metadata: {OUT_METADATA}")
    print(f"Rows: {len(compact):,}")
    print(f"Columns: {len(compact.columns):,}")
    print(f"Size MB: {OUT_METADATA.stat().st_size / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
