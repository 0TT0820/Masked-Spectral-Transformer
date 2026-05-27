"""Download raw Raman files from the MLROD ODR catalogue.

This script uses the file catalogue produced from the MLROD/ODR metadata and
downloads only raw Raman CSV files. It intentionally excludes CWT-transformed
files, XRD files, XRF files, and average spectra because those are not the raw
single-spectrum inputs used by the manuscript workflow.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "external_mlrod_berlanga_2022" / "catalog" / "mlrod_file_catalog.csv"
OUTDIR = ROOT / "data" / "external_mlrod_berlanga_2022" / "raw_raman_files"
MANIFEST = ROOT / "data" / "external_mlrod_berlanga_2022" / "raw_raman_download_manifest.csv"


RAW_KINDS = {"raw_training", "raw_labeled_test"}


SINGLE_MINERAL_TRAINING_USES = {"candidate_training_or_external_validation"}


def read_catalog() -> list[dict[str, str]]:
    with CATALOG.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("file_kind") in RAW_KINDS and r.get("download_url")]


def looks_like_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    head = path.read_bytes()[:65536].lower()
    if b"<html" in head or b"<!doctype" in head:
        return False
    return b"," in head and (b"label" in head or b"wave" in head or b"raman" in head)


def load_completed_manifest() -> set[str]:
    if not MANIFEST.exists():
        return set()
    completed: set[str] = set()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") in {"downloaded", "existing_csv"} and row.get("likely_csv") == "True":
                completed.add(row.get("file_id", ""))
    return completed


def append_manifest(row: dict[str, object]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    exists = MANIFEST.exists()
    fields = [
        "timestamp_utc",
        "file_id",
        "filename",
        "file_kind",
        "sample_id",
        "download_url",
        "status",
        "return_code",
        "bytes",
        "likely_csv",
        "message",
    ]
    with MANIFEST.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def download(row: dict[str, str], max_time: int, force: bool) -> dict[str, object]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / row["filename"]
    if out.exists() and out.stat().st_size > 1024 and looks_like_csv(out) and not force:
        return {
            "status": "existing_csv",
            "return_code": 0,
            "bytes": out.stat().st_size,
            "likely_csv": True,
            "message": "Existing file passed a lightweight CSV check.",
        }

    part = out.with_suffix(out.suffix + ".part")
    request = urllib.request.Request(
        row["download_url"],
        headers={"User-Agent": "Mozilla/5.0 (compatible; MST-Raman reproducibility script)"},
    )
    return_code = 0
    message = ""
    try:
        with urllib.request.urlopen(request, timeout=max_time) as response, part.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        part.replace(out)
    except Exception as exc:  # noqa: BLE001 - store network failures in the manifest.
        return_code = 1
        message = f"{type(exc).__name__}: {exc}"[:500]
    likely_csv = looks_like_csv(out)
    status = "downloaded" if return_code == 0 and likely_csv else "failed"
    return {
        "status": status,
        "return_code": return_code,
        "bytes": out.stat().st_size if out.exists() else 0,
        "likely_csv": likely_csv,
        "message": message,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of pending files to try.")
    parser.add_argument("--max-time", type=int, default=900, help="Per-file curl timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Re-download files that already look like CSV.")
    parser.add_argument(
        "--single-mineral-training-only",
        action="store_true",
        help="Download only MLROD single-mineral training files that map to manuscript classes.",
    )
    parser.add_argument("--sample-id-contains", default="", help="Optional case-insensitive sample-id substring filter.")
    args = parser.parse_args()

    rows = read_catalog()
    if args.single_mineral_training_only:
        rows = [
            r
            for r in rows
            if r.get("file_kind") == "raw_training"
            and r.get("project_superclass_mapping") not in {"mixture_or_rock", "rock_test_set", ""}
        ]
    if args.sample_id_contains:
        needle = args.sample_id_contains.lower()
        rows = [r for r in rows if needle in r.get("sample_id", "").lower()]
    rows = sorted(
        rows,
        key=lambda r: (
            float(r.get("approx_size_mb") or 1e9),
            r.get("sample_id", ""),
            r.get("filename", ""),
        ),
    )
    completed = load_completed_manifest()
    pending = [r for r in rows if args.force or r.get("file_id") not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Raw Raman files in catalogue: {len(rows)}")
    print(f"Already completed in manifest: {len(completed)}")
    print(f"Pending in this run: {len(pending)}")

    for idx, row in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] {row['file_id']} {row['filename']}")
        result = download(row, max_time=args.max_time, force=args.force)
        manifest_row = {
            "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "file_id": row.get("file_id", ""),
            "filename": row.get("filename", ""),
            "file_kind": row.get("file_kind", ""),
            "sample_id": row.get("sample_id", ""),
            "download_url": row.get("download_url", ""),
            **result,
        }
        append_manifest(manifest_row)
        print(
            f"    {result['status']} bytes={result['bytes']} "
            f"likely_csv={result['likely_csv']} rc={result['return_code']}"
        )


if __name__ == "__main__":
    main()
