from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "metadata" / "metadata_parent_945.csv"
DEFAULT_OUT = ROOT / "data" / "metadata" / "rruff_official_metadata.csv"
DEFAULT_ENRICHED = ROOT / "data" / "metadata" / "metadata_parent_945_rruff_enriched.csv"
DEFAULT_CACHE = ROOT / "data" / "metadata" / "rruff_cache"

RRUFF_BASE = "https://rruff.info"
USER_AGENT = "MST-Raman provenance enrichment for peer-reviewed reproducibility"


def compact(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()
    return text


def strip_html(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&#931;": "Σ",
        "&Sigma;": "Σ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return compact(text)


def get_after_label(text: str, label: str, stop_labels: list[str]) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)\s*(?={'|'.join(re.escape(s + ':') for s in stop_labels)}|$)"
    match = re.search(pattern, text, flags=re.I)
    return compact(match.group(1)) if match else ""


def parse_rruff_page(html: str, rruff_id: str, url: str) -> dict[str, str]:
    text = strip_html(html)
    fields = [
        "Name",
        "RRUFF ID",
        "Ideal Chemistry",
        "Locality",
        "Source",
        "Owner",
        "Description",
        "Status",
        "Mineral Group",
        "Measured Chemistry",
        "Wavelength",
        "Sample Description",
        "Instrument settings",
        "Resolution",
    ]
    stop_labels = fields + [
        "CHEMISTRY",
        "RAMAN SPECTRUM",
        "BROAD SCAN WITH SPECTRAL ARTIFACTS",
        "INFRARED SPECTRUM",
        "POWDER DIFFRACTION",
        "REFERENCES",
        "DOWNLOADS",
        "Quick search",
    ]
    parsed = {
        "rruff_id": rruff_id,
        "rruff_url": url,
        "rruff_fetch_status": "ok",
        "rruff_name": get_after_label(text, "Name", stop_labels),
        "rruff_ideal_chemistry": get_after_label(text, "Ideal Chemistry", stop_labels),
        "rruff_locality": get_after_label(text, "Locality", stop_labels),
        "rruff_source": get_after_label(text, "Source", stop_labels),
        "rruff_owner": get_after_label(text, "Owner", stop_labels),
        "rruff_description": get_after_label(text, "Description", stop_labels),
        "rruff_status": get_after_label(text, "Status", stop_labels),
        "rruff_mineral_group": get_after_label(text, "Mineral Group", stop_labels),
        "rruff_measured_chemistry": get_after_label(text, "Measured Chemistry", stop_labels),
        "rruff_raman_wavelengths": get_after_label(text, "Wavelength", stop_labels),
        "rruff_raman_sample_description": get_after_label(text, "Sample Description", stop_labels),
        "rruff_raman_instrument_settings": get_after_label(text, "Instrument settings", stop_labels),
        "rruff_raman_resolution": get_after_label(text, "Resolution", stop_labels),
    }
    if not parsed["rruff_name"] and re.search(rf"\b{re.escape(rruff_id)}\b", text):
        heading = re.search(rf"#?\s*([A-Za-z0-9 _\-/().]+)\s+{re.escape(rruff_id)}", text)
        if heading:
            parsed["rruff_name"] = compact(heading.group(1))
    return parsed


def fetch_url(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def candidate_urls(row: pd.Series) -> list[str]:
    rruff_id = str(row["source_id"]).strip()
    mineral = str(row.get("mineral_species", "") or row.get("subtype_label", "")).strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", mineral).strip("_").lower()
    urls = []
    urls.append(f"{RRUFF_BASE}/{rruff_id}")
    if slug:
        urls.append(f"{RRUFF_BASE}/{slug}/display%3Ddefault/{rruff_id}")
    urls.append(f"{RRUFF_BASE}/export/{rruff_id}")
    urls.append(f"{RRUFF_BASE}/export1538/{rruff_id}")
    return list(dict.fromkeys(urls))


def fetch_record(row: pd.Series, cache_dir: Path, sleep_s: float, refresh: bool) -> dict[str, str]:
    rruff_id = str(row["source_id"]).strip()
    cache_file = cache_dir / f"{rruff_id}.html"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    if cache_file.exists() and not refresh:
        html = cache_file.read_text(encoding="utf-8", errors="replace")
        parsed = parse_rruff_page(html, rruff_id, str(row.get("rruff_url", "")))
        parsed["rruff_fetch_status"] = "ok_cached"
        return parsed

    for url in candidate_urls(row):
        try:
            html = fetch_url(url)
            if rruff_id in html:
                cache_file.write_text(html, encoding="utf-8")
                time.sleep(sleep_s)
                return parse_rruff_page(html, rruff_id, url)
            errors.append(f"{url}: id not found in response")
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
        time.sleep(sleep_s)

    return {
        "rruff_id": rruff_id,
        "rruff_url": "",
        "rruff_fetch_status": "failed",
        "rruff_fetch_error": " | ".join(errors)[:1000],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch official RRUFF sample metadata and merge it with the parent Raman metadata table."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--enriched-out", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, encoding="utf-8-sig")
    rruff_rows = metadata[metadata["source_type"].eq("RRUFF database")].copy()
    rruff_unique = (
        rruff_rows.sort_values(["source_id", "mineral_species", "subtype_label"])
        .drop_duplicates("source_id")
        .reset_index(drop=True)
    )
    if args.limit:
        rruff_unique = rruff_unique.head(args.limit)

    records = []
    total = len(rruff_unique)
    for index, row in rruff_unique.iterrows():
        rruff_id = row["source_id"]
        print(f"[{index + 1}/{total}] {rruff_id}", flush=True)
        record = fetch_record(row, args.cache_dir, args.sleep, args.refresh)
        records.append(record)

    official = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    official.to_csv(args.out, index=False, encoding="utf-8-sig")

    enriched = metadata.merge(official, left_on="source_id", right_on="rruff_id", how="left")
    enriched.to_csv(args.enriched_out, index=False, encoding="utf-8-sig")

    summary = {
        "metadata": str(args.metadata),
        "rruff_unique_ids": int(len(rruff_unique)),
        "fetched_ok": int(official["rruff_fetch_status"].astype(str).str.startswith("ok").sum())
        if not official.empty and "rruff_fetch_status" in official
        else 0,
        "failed": int((official.get("rruff_fetch_status", pd.Series(dtype=str)) == "failed").sum()),
        "official_metadata": str(args.out),
        "enriched_metadata": str(args.enriched_out),
    }
    (args.out.parent / "rruff_official_metadata_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
