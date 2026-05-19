"""Build training-ready metadata for public Martian meteorite Raman supplements.

This script converts public Mendeley Data Raman files into the repository's
one-spectrum-per-row metadata format. Only spectra with explicit mineral labels
are marked as supervised training records. Unlabeled spectra are retained as
candidate records for provenance and future manual curation, but are not merged
into the training-ready table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external_martian_meteorite_sources"
SPECTRA_OUT = ROOT / "data" / "spectra" / "martian_meteorite_mendeley"
META_OUT = ROOT / "data" / "metadata"

BASE_TRAINING = META_OUT / "metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv"


SUPERVISED_DATASET = {
    "dataset_id": "c6t3v22x2x",
    "doi": "10.17632/c6t3v22x2x.1",
    "url": "https://data.mendeley.com/datasets/c6t3v22x2x/1",
    "title": (
        "IRON AND TITANIUM SKELETAL STRUCTURES IN NAKHLITES MIL 090030, "
        "MIL 090136, MIL 090032, AND MIL 03346: COMPARATIVE ANALYSIS WITH "
        "TERRESTRIAL ANALOGUES FROM CANARY ISLANDS, SPAIN"
    ),
    "publication_date": "2026-01-08",
    "license": "CC BY 4.0",
    "contributors": "Leire Coloma; Julene Aramendia; Fernando Alberquilla; Gorka Arana; Juan Manuel Madariaga",
    "sample_provenance": "Martian paired nakhlites MIL 090030, MIL 090136, MIL 090032, and MIL 03346",
    "reference": (
        "Coloma, L., Aramendia, J., Alberquilla, F., Arana, G., & Madariaga, J. M. "
        "(2026). Iron and titanium skeletal structures in nakhlites MIL 090030, "
        "MIL 090136, MIL 090032, and MIL 03346: Comparative analysis with "
        "terrestrial analogues from Canary Islands, Spain. Mendeley Data, V1. "
        "https://doi.org/10.17632/c6t3v22x2x.1"
    ),
}


UNLABELED_DATASET = {
    "dataset_id": "97hjg7hcft",
    "doi": "10.17632/97hjg7hcft.1",
    "url": "https://data.mendeley.com/datasets/97hjg7hcft/1",
    "title": (
        "COMPARING THE INORGANIC AND ORGANIC COMPOSITION OF THE MIL 090030, "
        "MIL 090136, MIL 090032 AND MIL 03346 MARTIAN PAIRED METEORITES: ARE ALL OF THEM PAIRED?"
    ),
    "publication_date": "2026-04-15",
    "license": "CC BY 4.0",
    "sample_provenance": "Martian paired meteorites MIL 090030, MIL 090136, MIL 090032, and MIL 03346",
    "reference": (
        "Coloma, L., Aramendia, J., Alberquilla, F., Arana, G., & Madariaga, J. M. "
        "(2026). Comparing the inorganic and organic composition of the MIL 090030, "
        "MIL 090136, MIL 090032 and MIL 03346 Martian paired meteorites: Are all of "
        "them paired? Mendeley Data, V1. https://doi.org/10.17632/97hjg7hcft.1"
    ),
}


MINERAL_TO_SUPERCLASS = {
    "ilmenite": "Oxides/Hydroxides",
    "magnetite": "Oxides/Hydroxides",
    "titanomagnetite": "Oxides/Hydroxides",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_two_column_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None, names=["raman_shift_cm-1", "intensity"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.drop_duplicates(subset=["raman_shift_cm-1"]).sort_values("raman_shift_cm-1")
    return df


def build_supervised_records() -> pd.DataFrame:
    raw_dir = EXTERNAL / "mendeley_c6t3v22x2x_raw"
    SPECTRA_OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    file_manifest = json.loads((EXTERNAL / "mendeley_c6t3v22x2x_files_root.json").read_text(encoding="utf-8"))
    manifest_by_name = {row["filename"]: row for row in file_manifest}

    for raw_path in sorted(raw_dir.glob("*.txt")):
        mineral = raw_path.stem
        major_category = MINERAL_TO_SUPERCLASS.get(mineral.lower())
        if not major_category:
            continue

        spectrum_id = f"MARTIAN_METEORITE_MENDELEY_C6T3V22X2X_{mineral.upper()}"
        out_name = f"{spectrum_id}.csv"
        out_path = SPECTRA_OUT / out_name
        spec = read_two_column_txt(raw_path)
        spec.to_csv(out_path, index=False)

        file_info = manifest_by_name.get(raw_path.name, {})
        content_details = file_info.get("content_details", {})

        records.append(
            {
                "file_name": out_name,
                "group_label": major_category,
                "subtype_label": mineral.capitalize() if mineral != "titanomagnetite" else "Titanomagnetite",
                "Mg#": pd.NA,
                "Fe#": pd.NA,
                "Ca#": pd.NA,
                "Na#": pd.NA,
                "K#": pd.NA,
                "wave": pd.NA,
                "formula": pd.NA,
                "major_category": major_category,
                "file_name_clean": out_name.replace(".csv", ""),
                "file_path": str(out_path.resolve()),
                "match_method": "public_mendeley_dataset",
                "file_exists": True,
                "spectrum_id": spectrum_id,
                "parsed_file_name": out_name,
                "mineral_species": mineral.capitalize() if mineral != "titanomagnetite" else "Titanomagnetite",
                "source_id": f"Mendeley:{SUPERVISED_DATASET['dataset_id']}:{raw_path.name}",
                "source_type": "Martian meteorite spectra",
                "spectrum_type": "Raman",
                "excitation_nm": pd.NA,
                "instrument": "Raman instrument not specified in downloaded file; verify from associated article if needed",
                "data_level": "public raw two-column spectrum",
                "orientation": pd.NA,
                "sample_provenance": SUPERVISED_DATASET["sample_provenance"],
                "measurement_conditions": "Public Mendeley Data Raman text file; wavenumber and intensity provided as two columns",
                "label_basis": "Mineral label from public Raman file name in Mendeley Data",
                "reference": SUPERVISED_DATASET["reference"],
                "source_note": f"{SUPERVISED_DATASET['title']}; license {SUPERVISED_DATASET['license']}",
                "spectral_min_cm-1": float(spec["raman_shift_cm-1"].min()),
                "spectral_max_cm-1": float(spec["raman_shift_cm-1"].max()),
                "n_original_points": int(len(spec)),
                "spectral_range_cm-1": f"{spec['raman_shift_cm-1'].min():.1f}-{spec['raman_shift_cm-1'].max():.1f}",
                "file_sha256": sha256_file(out_path),
                "parent_group": f"Mendeley Data {SUPERVISED_DATASET['doi']}",
                "preprocessing_planned": (
                    "Parse wavenumber/intensity columns; sort and deduplicate wavenumber axis; "
                    "interpolate to common grid; baseline correction; nonnegative clipping; "
                    "max-intensity normalization; first derivative channel"
                ),
                "augmentation_used": "no",
                "qc_status": "include_public_martian_meteorite_supplement",
                "qc_reason": "Explicit mineral label is present in file name; public DOI and CC BY 4.0 license available",
                "recommended_action": "retain_after_manual_qc",
                "split_main": "train",
                "split_zero_shot_protocol": "earth_train_pool",
                "mendeley_dataset_id": SUPERVISED_DATASET["dataset_id"],
                "mendeley_doi": SUPERVISED_DATASET["doi"],
                "mendeley_url": SUPERVISED_DATASET["url"],
                "mendeley_license": SUPERVISED_DATASET["license"],
                "mendeley_file_id": file_info.get("id", pd.NA),
                "mendeley_file_sha256": content_details.get("sha256_hash", pd.NA),
                "training_label_usable": True,
            }
        )

    return pd.DataFrame.from_records(records)


def build_unlabeled_candidate_records() -> pd.DataFrame:
    xlsx_path = EXTERNAL / "mendeley_97hjg7hcft_Raman_spectra.xlsx"
    if not xlsx_path.exists():
        return pd.DataFrame()
    df = pd.read_excel(xlsx_path, sheet_name="Minerals")
    records: list[dict[str, object]] = []
    candidate_dir = SPECTRA_OUT / "unlabeled_candidates_97hjg7hcft"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    pair_index = 0
    for i in range(0, len(df.columns), 2):
        wave_col = df.columns[i]
        intensity_col = df.columns[i + 1]
        spec = pd.DataFrame(
            {
                "raman_shift_cm-1": pd.to_numeric(df[wave_col], errors="coerce"),
                "intensity": pd.to_numeric(df[intensity_col], errors="coerce"),
            }
        ).dropna()
        if spec.empty:
            continue
        spec = spec.drop_duplicates(subset=["raman_shift_cm-1"]).sort_values("raman_shift_cm-1")
        spectrum_id = f"MARTIAN_METEORITE_MENDELEY_97HJG7HCFT_UNLABELED_{pair_index:02d}"
        out_path = candidate_dir / f"{spectrum_id}.csv"
        spec.to_csv(out_path, index=False)
        records.append(
            {
                "spectrum_id": spectrum_id,
                "file_name": out_path.name,
                "file_path": str(out_path.resolve()),
                "source_type": "Martian meteorite spectra",
                "sample_provenance": UNLABELED_DATASET["sample_provenance"],
                "mendeley_dataset_id": UNLABELED_DATASET["dataset_id"],
                "mendeley_doi": UNLABELED_DATASET["doi"],
                "mendeley_url": UNLABELED_DATASET["url"],
                "mendeley_license": UNLABELED_DATASET["license"],
                "reference": UNLABELED_DATASET["reference"],
                "spectral_min_cm-1": float(spec["raman_shift_cm-1"].min()),
                "spectral_max_cm-1": float(spec["raman_shift_cm-1"].max()),
                "n_original_points": int(len(spec)),
                "file_sha256": sha256_file(out_path),
                "training_label_usable": False,
                "candidate_status": "not_added_to_supervised_training_no_per_spectrum_mineral_label",
                "candidate_reason": (
                    "Downloaded workbook contains paired Wave/Intensity columns, but no per-column mineral label "
                    "or point/sample assignment was present in the file."
                ),
            }
        )
        pair_index += 1

    return pd.DataFrame.from_records(records)


def write_summary(supervised: pd.DataFrame, candidates: pd.DataFrame, combined: pd.DataFrame) -> None:
    source_counts = (
        combined.groupby("source_type", dropna=False)
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["source_type"])
    )
    category_counts = (
        combined.groupby("major_category", dropna=False)
        .size()
        .reset_index(name="n_spectra")
        .sort_values(["major_category"])
    )
    summary = [
        "# Martian Meteorite Mendeley Supplement",
        "",
        f"Supervised spectra added: {len(supervised)}",
        f"Unlabeled candidate spectra retained but not added to training: {len(candidates)}",
        f"Combined training-ready metadata rows after supervised supplement: {len(combined)}",
        "",
        "## Supervised Added Records",
        "",
        supervised[["spectrum_id", "mineral_species", "major_category", "mendeley_doi", "n_original_points"]].to_markdown(index=False)
        if not supervised.empty
        else "None.",
        "",
        "## Source Counts In Combined Training-Ready Table",
        "",
        source_counts.to_markdown(index=False),
        "",
        "## Mineral Superclass Counts In Combined Training-Ready Table",
        "",
        category_counts.to_markdown(index=False),
    ]
    (META_OUT / "martian_meteorite_mendeley_supplement_summary.md").write_text("\n".join(summary), encoding="utf-8")
    source_counts.to_csv(META_OUT / "martian_meteorite_mendeley_combined_source_counts.csv", index=False)
    category_counts.to_csv(META_OUT / "martian_meteorite_mendeley_combined_category_counts.csv", index=False)


def main() -> None:
    META_OUT.mkdir(parents=True, exist_ok=True)
    supervised = build_supervised_records()
    candidates = build_unlabeled_candidate_records()
    supervised.to_csv(META_OUT / "metadata_martian_meteorite_mendeley_supervised_supplement.csv", index=False)
    candidates.to_csv(META_OUT / "metadata_martian_meteorite_mendeley_unlabeled_candidates.csv", index=False)

    base = pd.read_csv(BASE_TRAINING)
    for col in supervised.columns:
        if col not in base.columns:
            base[col] = pd.NA
    for col in base.columns:
        if col not in supervised.columns:
            supervised[col] = pd.NA
    combined = pd.concat([base, supervised[base.columns]], ignore_index=True)
    combined.to_csv(META_OUT / "metadata_training_ready_plus_martian_meteorite_mendeley.csv", index=False)
    write_summary(supervised, candidates, combined)

    print(f"supervised_added={len(supervised)}")
    print(f"unlabeled_candidates={len(candidates)}")
    print(f"combined_training_ready_rows={len(combined)}")


if __name__ == "__main__":
    main()
