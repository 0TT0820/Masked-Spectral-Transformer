"""Build the review-ready training database with an explicit DUV library.

This script consolidates the current supervised parent database with the
PDS-traceable SHERLOC SaU 008 calibration-target spectra. It does not overwrite
legacy metadata. Instead it writes versioned tables that distinguish:

- supervised Earth/laboratory reference spectra,
- labeled SHERLOC in-situ spectra for fine-tuning or validation,
- unlabeled/bulk SHERLOC calibration-target spectra for domain adaptation.

This separation is important because SHERLOC calibration products such as
SaU 008 are mission-domain DUV spectra, but the PDS RRS files do not provide
point-level mineral labels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = ROOT / "data" / "metadata"
OVERVIEW_DIR = ROOT / "data" / "overview" / "training_database_v2"
DOCS_DIR = ROOT / "docs"

BASE_WITH_SAU = METADATA_DIR / "metadata_training_ready_plus_sau008_domain_adaptation.csv"
BASE_WITHOUT_SAU = METADATA_DIR / "metadata_training_ready_plus_martian_meteorite_mendeley.csv"
SAU_SCRIPT = ROOT / "src" / "build_sherloc_sau008_calibration_dataset.py"

ALL_OUTPUT = METADATA_DIR / "metadata_training_database_v2_all_sources.csv"
DUV_OUTPUT = METADATA_DIR / "metadata_duv_training_library_v1.csv"


SHERLOC_SOURCE_ALIASES = {
    "SHERLOC in-situ spectra": "SHERLOC in-situ Mars 2020",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_source_type(value: object) -> str:
    text = clean_text(value)
    return SHERLOC_SOURCE_ALIASES.get(text, text)


def final_label(row: pd.Series) -> str:
    for col in ("paper_table1_superclass", "major_category", "group_label"):
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def final_species(row: pd.Series) -> str:
    for col in ("mineral_species", "subtype_label", "mineral_label"):
        value = clean_text(row.get(col, ""))
        if value:
            return value
    return ""


def boolish(value: object) -> bool:
    return clean_text(value).lower() in {"true", "1", "yes", "y"}


def source_domain(row: pd.Series) -> str:
    source = row["source_type_normalized"]
    excitation = clean_text(row.get("excitation_nm", ""))
    if source == "SHERLOC calibration target Mars meteorite SaU 008":
        return "sherloc_calibration_duv"
    if source == "SHERLOC in-situ Mars 2020":
        return "sherloc_insitu_duv"
    if source == "Lab-acquired DUV spectra" or excitation == "248.6":
        return "laboratory_duv"
    if source == "RRUFF database":
        return "visible_nir_reference"
    if source == "Martian meteorite spectra":
        return "meteorite_reference"
    return "other"


def training_role(row: pd.Series) -> str:
    source = row["source_type_normalized"]
    split = clean_text(row.get("split_main", ""))
    if source == "SHERLOC calibration target Mars meteorite SaU 008":
        return "duv_domain_adaptation_bulk_unlabeled"
    if source == "SHERLOC in-situ Mars 2020" and split == "external_sherloc":
        return "duv_sherloc_labeled_external_validation"
    if source == "SHERLOC in-situ Mars 2020":
        return "duv_sherloc_labeled_finetune_pool"
    if source == "Lab-acquired DUV spectra":
        return "duv_supervised_reference"
    if source == "RRUFF database":
        return "visible_nir_supervised_reference"
    if source == "Martian meteorite spectra":
        return "meteorite_supervised_reference"
    return "review_before_training"


def supervised_label_usable(row: pd.Series) -> bool:
    if row["training_role"] == "duv_domain_adaptation_bulk_unlabeled":
        return False
    if not row["label_category_final"]:
        return False
    if boolish(row.get("sherloc_training_label_usable", "")):
        return True
    if clean_text(row.get("source_type_normalized", "")) in {
        "RRUFF database",
        "Lab-acquired DUV spectra",
        "Martian meteorite spectra",
        "SHERLOC in-situ Mars 2020",
    }:
        return True
    return boolish(row.get("training_label_usable", ""))


def duv_library_include(row: pd.Series) -> bool:
    source = row["source_type_normalized"]
    if source in {
        "Lab-acquired DUV spectra",
        "SHERLOC in-situ Mars 2020",
        "SHERLOC calibration target Mars meteorite SaU 008",
    }:
        return True
    return clean_text(row.get("excitation_nm", "")) == "248.6"


def split_v2(row: pd.Series) -> str:
    role = row["training_role"]
    if role == "duv_domain_adaptation_bulk_unlabeled":
        return "domain_adaptation_only"
    if role == "duv_sherloc_labeled_finetune_pool":
        return "sherloc_finetune_pool"
    if role == "duv_sherloc_labeled_external_validation":
        return "sherloc_external_validation"
    value = clean_text(row.get("split_main", ""))
    return value or "unspecified"


def write_counts(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "value", "count"])
        for field in [
            "source_type_normalized",
            "source_domain",
            "training_role",
            "split_v2",
            "label_category_final",
            "supervised_label_usable_v2",
            "duv_library_include",
        ]:
            counts: dict[str, int] = {}
            for row in rows:
                key = str(row.get(field, ""))
                counts[key] = counts.get(key, 0) + 1
            for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                writer.writerow([field, key, count])


def write_cross_table(path: Path, df: pd.DataFrame, index: str, columns: str) -> None:
    table = pd.crosstab(df[index].fillna("").astype(str), df[columns].fillna("").astype(str))
    table.to_csv(path)


def write_doc(all_df: pd.DataFrame, duv_df: pd.DataFrame) -> None:
    doc = f"""# Training Database v2

This versioned metadata set reorganizes the Raman training data after adding
PDS-traceable SHERLOC DUV spectra. It keeps a clear distinction between
supervised labels and DUV spectra that are useful only for domain adaptation or
manual review.

## Files

- `data/metadata/metadata_training_database_v2_all_sources.csv`
- `data/metadata/metadata_duv_training_library_v1.csv`
- `data/overview/training_database_v2/all_sources_counts.csv`
- `data/overview/training_database_v2/duv_library_counts.csv`
- `data/overview/training_database_v2/duv_source_by_label.csv`
- `data/overview/training_database_v2/all_source_by_training_role.csv`

## Counts

- All-source metadata rows: {len(all_df)}
- DUV-library rows: {len(duv_df)}
- DUV rows usable for supervised labels: {int(duv_df['supervised_label_usable_v2'].sum())}
- DUV rows reserved for domain adaptation or manual review: {int((~duv_df['supervised_label_usable_v2']).sum())}

## Interpretation

The DUV library includes laboratory DUV spectra, labeled SHERLOC in-situ Mars
2020 spectra, and SHERLOC SaU 008 calibration-target spectra. SaU 008 spectra
are included because they are real SHERLOC DUV measurements of a Martian
meteorite calibration target, but their PDS products do not provide point-level
mineral labels. They are therefore marked as `duv_domain_adaptation_bulk_unlabeled`
and excluded from closed-set supervised mineral classification unless later
manual point-level labels are added.

The Caltech/JPL SHERLOC-analog 62-mineral library is documented separately in
`docs/caltech_sherloc_duv_62min_import_status.md`. It is not imported into the
numeric training table because the public supplement located so far contains
plotted spectra rather than machine-readable wavenumber-intensity arrays.
"""
    (DOCS_DIR / "training_database_v2.md").write_text(doc, encoding="utf-8")


def main() -> None:
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    input_path = BASE_WITH_SAU if BASE_WITH_SAU.exists() else BASE_WITHOUT_SAU
    df = pd.read_csv(input_path, low_memory=False)
    df["source_type_normalized"] = df.get("source_type", "").apply(normalize_source_type)
    df["label_category_final"] = df.apply(final_label, axis=1)
    df["mineral_species_final"] = df.apply(final_species, axis=1)
    df["source_domain"] = df.apply(source_domain, axis=1)
    df["training_role"] = df.apply(training_role, axis=1)
    df["supervised_label_usable_v2"] = df.apply(supervised_label_usable, axis=1)
    df["duv_library_include"] = df.apply(duv_library_include, axis=1)
    df["split_v2"] = df.apply(split_v2, axis=1)

    sort_cols = ["source_domain", "source_type_normalized", "training_role", "label_category_final", "spectrum_id"]
    df = df.sort_values([col for col in sort_cols if col in df.columns]).reset_index(drop=True)
    duv_df = df[df["duv_library_include"]].copy().reset_index(drop=True)

    df.to_csv(ALL_OUTPUT, index=False)
    duv_df.to_csv(DUV_OUTPUT, index=False)

    write_counts(OVERVIEW_DIR / "all_sources_counts.csv", df.to_dict("records"))
    write_counts(OVERVIEW_DIR / "duv_library_counts.csv", duv_df.to_dict("records"))
    write_cross_table(OVERVIEW_DIR / "duv_source_by_label.csv", duv_df, "source_type_normalized", "label_category_final")
    write_cross_table(OVERVIEW_DIR / "all_source_by_training_role.csv", df, "source_type_normalized", "training_role")
    write_doc(df, duv_df)


if __name__ == "__main__":
    main()
