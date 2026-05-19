"""Model comparison and confidence thresholds on labeled SHERLOC in-situ spectra.

This script complements the reference-domain benchmark. It pools all labeled
SHERLOC in-situ spectra, performs repeated random train/validation splits, and
compares the reviewer-requested model families under the same split. This is a
within-domain SHERLOC random-split validation, not an independent target or
region transfer test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from train_model_comparison import (
    PLSDA,
    RamanDataset,
    StandardTransformer,
    MaskedSpectralTransformer,
    evaluate_arrays,
    fix_seed,
    flatten_features,
    predict_torch,
    train_torch_model,
)
from run_model_selection import TunedCNN
from run_review_updated_training_v2 import METADATA_V2, load_v2_metadata, sherloc_split
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "sherloc_in_situ_model_comparison_v3"
SEED = 2024
THRESHOLDS = np.round(np.arange(0.0, 0.951, 0.05), 2)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_pooled_sherloc(metadata_file: Path) -> pd.DataFrame:
    full = load_v2_metadata(metadata_file)
    ft, ext = sherloc_split(full)
    ft["pooled_origin"] = "previous_finetune_pool"
    ext["pooled_origin"] = "previous_external_validation"
    pooled = pd.concat([ft, ext], ignore_index=True)
    return pooled.reset_index(drop=True)


def split_indices(y: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray, str]:
    counts = pd.Series(y).value_counts()
    singleton_labels = set(counts[counts < 2].index.tolist())
    singleton_idx = np.array([i for i, yy in enumerate(y) if yy in singleton_labels], dtype=int)
    normal_idx = np.array([i for i, yy in enumerate(y) if yy not in singleton_labels], dtype=int)
    if len(normal_idx) == 0:
        idx = np.arange(len(y))
        train_idx, val_idx = train_test_split(idx, test_size=val_fraction, random_state=seed)
        return np.sort(train_idx), np.sort(val_idx), "random_no_stratification"
    train_norm, val_idx = train_test_split(
        normal_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=y[normal_idx],
    )
    train_idx = np.concatenate([train_norm, singleton_idx])
    return np.sort(train_idx), np.sort(val_idx), "stratified_for_labels_with_n>=2_singletons_forced_to_train"


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict[str, object]:
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1_all_sherloc_labels": float(
            f1_score(y_true, pred, labels=np.arange(len(classes)), average="macro", zero_division=0)
        ),
        "present_label_macro_f1": float(f1_score(y_true, pred, labels=np.unique(y_true), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "predicted_labels": ";".join(sorted({classes[i] for i in np.unique(pred)})),
    }


def threshold_rows(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    model: str,
    dataset: str,
) -> list[dict[str, object]]:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    n = len(y_true)
    for threshold in THRESHOLDS:
        accepted = conf >= threshold
        accepted_n = int(accepted.sum())
        wrong_accepted = accepted & (pred != y_true)
        correct_accepted = accepted & (pred == y_true)
        if accepted_n:
            acc = float(np.mean(pred[accepted] == y_true[accepted]))
            macro_f1 = float(
                f1_score(
                    y_true[accepted],
                    pred[accepted],
                    labels=np.unique(y_true[accepted]),
                    average="macro",
                    zero_division=0,
                )
            )
            fdr = float(wrong_accepted.sum() / accepted_n)
        else:
            acc = np.nan
            macro_f1 = np.nan
            fdr = np.nan

        macro_fprs = []
        for class_id in range(len(classes)):
            fp = int(np.sum(accepted & (pred == class_id) & (y_true != class_id)))
            tn = int(np.sum((y_true != class_id) & (~accepted | (pred != class_id))))
            denom = fp + tn
            macro_fprs.append(fp / denom if denom else np.nan)

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "threshold": float(threshold),
                "n_total": int(n),
                "accepted_n": accepted_n,
                "rejected_n": int((~accepted).sum()),
                "coverage": float(accepted_n / n) if n else np.nan,
                "accuracy_on_accepted": acc,
                "macro_f1_on_accepted_present_labels": macro_f1,
                "operational_recall_correct_accepted_over_all": float(correct_accepted.sum() / n) if n else np.nan,
                "false_discovery_rate_wrong_among_accepted": fdr,
                "false_positive_rate_wrong_accepted_over_all": float(wrong_accepted.sum() / n) if n else np.nan,
                "macro_one_vs_rest_fpr": float(np.nanmean(macro_fprs)),
            }
        )
    return rows


def fit_predict_sklearn(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    tune_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    if model_name == "pca_svm":
        candidates = []
        for stride in [4, 8]:
            for n_pca in [20, 40, 80]:
                max_pca = min(n_pca, len(train_idx) - 1, flatten_features(x[train_idx], stride).shape[1])
                if max_pca < 2:
                    continue
                for c_value in [1.0, 10.0]:
                    for gamma in ["scale", 0.001]:
                        model = Pipeline(
                            [
                                ("scale", StandardScaler()),
                                ("pca", PCA(n_components=max_pca, random_state=SEED)),
                                (
                                    "svm",
                                    SVC(
                                        kernel="rbf",
                                        C=c_value,
                                        gamma=gamma,
                                        class_weight="balanced",
                                        probability=True,
                                        random_state=SEED,
                                    ),
                                ),
                            ]
                        )
                        model.fit(flatten_features(x[train_idx], stride), y[train_idx])
                        tune_probs = model.predict_proba(flatten_features(x[tune_idx], stride))
                        tune_score = f1_score(y[tune_idx], np.argmax(tune_probs, axis=1), average="macro", zero_division=0)
                        candidates.append((tune_score, {"stride": stride, "pca": max_pca, "C": c_value, "gamma": gamma}, model))
        best_score, params, model = max(candidates, key=lambda item: item[0])
        probs = model.predict_proba(flatten_features(x[val_idx], int(params["stride"])))
        return {"params": json.dumps(params), "internal_tune_macro_f1": float(best_score)}, probs

    if model_name == "pls_da":
        candidates = []
        for stride in [4, 8]:
            for n_components in [2, 4, 6]:
                max_components = min(n_components, len(np.unique(y[train_idx])) - 1, len(train_idx) - 1)
                if max_components < 2:
                    continue
                model = PLSDA(n_components=max_components)
                model.fit(flatten_features(x[train_idx], stride), y[train_idx])
                tune_probs = model.predict_proba(flatten_features(x[tune_idx], stride))
                tune_score = f1_score(y[tune_idx], np.argmax(tune_probs, axis=1), average="macro", zero_division=0)
                candidates.append((tune_score, {"stride": stride, "components": max_components}, model))
        best_score, params, model = max(candidates, key=lambda item: item[0])
        probs = model.predict_proba(flatten_features(x[val_idx], int(params["stride"])))
        return {"params": json.dumps(params), "internal_tune_macro_f1": float(best_score)}, probs

    raise ValueError(model_name)


def train_eval_torch_trial(
    model_key: str,
    trial_suffix: str,
    model: torch.nn.Module,
    params: dict[str, object],
    x: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    tune_idx: np.ndarray,
    val_idx: np.ndarray,
    classes: list[str],
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray, dict[str, object]]:
    train_ds = RamanDataset(x[train_idx], masks[train_idx], y[train_idx], augment=False)
    tune_ds = RamanDataset(x[tune_idx], masks[tune_idx], y[tune_idx], augment=False)
    val_ds = RamanDataset(x[val_idx], masks[val_idx], y[val_idx], augment=False)
    trial_name = f"seed{seed}_{trial_suffix}"
    metrics = train_torch_model(
        trial_name,
        model,
        train_ds,
        tune_ds,
        val_ds,
        classes,
        out_dir,
        epochs,
        batch_size,
        float(params["lr"]),
        device,
        augment=False,
    )
    checkpoint = out_dir / f"{trial_name}.pth"
    reloaded = make_torch_model(model_key, len(classes), params).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    probs, _ = predict_torch(reloaded, DataLoader(val_ds, batch_size=batch_size, shuffle=False), device)
    return float(metrics["best_val_macro_f1"]), probs, params


def make_torch_model(model_key: str, n_classes: int, params: dict[str, object]) -> torch.nn.Module:
    if model_key == "cnn":
        return TunedCNN(num_classes=n_classes, dropout=float(params["dropout"]))
    if model_key == "standard_transformer":
        return StandardTransformer(
            num_classes=n_classes,
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch"]),
        )
    if model_key == "mst":
        return MaskedSpectralTransformer(
            num_classes=n_classes,
            d_model=int(params["d_model"]),
            layers=int(params["layers"]),
            patch_size=int(params["patch"]),
        )
    raise ValueError(model_key)


def fit_predict_torch_family(
    model_key: str,
    x: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    tune_idx: np.ndarray,
    val_idx: np.ndarray,
    classes: list[str],
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray]:
    if model_key == "cnn":
        trials = [
            {"lr": 1e-3, "dropout": 0.25},
            {"lr": 3e-4, "dropout": 0.40},
        ]
    elif model_key == "standard_transformer":
        trials = [
            {"lr": 1e-4, "d_model": 96, "layers": 3, "patch": 8},
            {"lr": 3e-5, "d_model": 128, "layers": 4, "patch": 8},
        ]
    elif model_key == "mst":
        trials = [
            {"lr": 1e-4, "d_model": 96, "layers": 3, "patch": 8},
            {"lr": 3e-5, "d_model": 128, "layers": 4, "patch": 8},
            {"lr": 3e-5, "d_model": 128, "layers": 4, "patch": 4},
        ]
    else:
        raise ValueError(model_key)

    rows = []
    for i, params in enumerate(trials, start=1):
        model = make_torch_model(model_key, len(classes), params)
        score, probs, used_params = train_eval_torch_trial(
            model_key,
            f"{model_key}_trial{i}",
            model,
            params,
            x,
            masks,
            y,
            train_idx,
            tune_idx,
            val_idx,
            classes,
            out_dir,
            seed,
            epochs,
            batch_size,
            device,
        )
        rows.append((score, probs, used_params))
    best_score, best_probs, best_params = max(rows, key=lambda item: item[0])
    return {"params": json.dumps(best_params), "internal_tune_macro_f1": float(best_score)}, best_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on pooled labeled SHERLOC in-situ spectra.")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--variant", default="despike_sg11_asls")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--tune-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch_dir = args.out_dir / "torch"
    torch_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "_cache"

    pooled = make_pooled_sherloc(args.metadata_file)
    pooled.to_csv(args.out_dir / "pooled_sherloc_labeled_samples.csv", index=False, encoding="utf-8-sig")
    encoder = LabelEncoder()
    pooled["label_id"] = encoder.fit_transform(pooled["model_label"])
    classes = list(encoder.classes_)
    x, masks, stats = build_sherloc_arrays(pooled, cache_dir, "pooled_labeled_sherloc", args.variant, args.refresh_cache)
    y = pooled["label_id"].to_numpy(dtype=np.int64)
    stats.to_csv(args.out_dir / "pooled_sherloc_preprocessing_stats.csv", index=False, encoding="utf-8-sig")

    rows = []
    prediction_frames = []
    for seed in args.seeds:
        log(f"SHERLOC split seed {seed}")
        train_pool_idx, val_idx, split_note = split_indices(y, seed, args.val_fraction)
        train_idx, tune_idx, tune_note = split_indices(y[train_pool_idx], seed + 17, args.tune_fraction)
        train_idx = train_pool_idx[train_idx]
        tune_idx = train_pool_idx[tune_idx]

        split_df = pooled.copy()
        split_df["split_seed"] = seed
        split_df["sherloc_random_split"] = "unused"
        split_df.loc[train_idx, "sherloc_random_split"] = "train"
        split_df.loc[tune_idx, "sherloc_random_split"] = "internal_tune"
        split_df.loc[val_idx, "sherloc_random_split"] = "heldout_validation"
        split_df.to_csv(args.out_dir / f"sherloc_random_split_seed_{seed}.csv", index=False, encoding="utf-8-sig")

        model_outputs: list[tuple[str, dict[str, object], np.ndarray]] = []
        for model_name in ["pca_svm", "pls_da"]:
            log(f"{model_name} seed {seed}")
            meta, probs = fit_predict_sklearn(model_name, x, y, train_idx, tune_idx, val_idx)
            model_outputs.append((model_name, meta, probs))
        for model_name in ["cnn", "standard_transformer", "mst"]:
            log(f"{model_name} seed {seed}")
            meta, probs = fit_predict_torch_family(
                model_name,
                x,
                masks,
                y,
                train_idx,
                tune_idx,
                val_idx,
                classes,
                torch_dir,
                seed,
                args.epochs,
                args.batch_size,
                device,
            )
            model_outputs.append((model_name, meta, probs))

        for model_name, meta, probs in model_outputs:
            result = {
                "model": model_name,
                "seed": seed,
                "variant": args.variant,
                "train_n": int(len(train_idx)),
                "internal_tune_n": int(len(tune_idx)),
                "heldout_validation_n": int(len(val_idx)),
                "split_note": split_note,
                "internal_tune_note": tune_note,
                **meta,
                **metrics_from_probs(y[val_idx], probs, classes),
            }
            rows.append(result)
            pred = np.argmax(probs, axis=1)
            conf = np.max(probs, axis=1)
            pred_df = pooled.iloc[val_idx][["spectrum_id", "model_label", "pooled_origin"]].copy()
            pred_df["seed"] = seed
            pred_df["model"] = model_name
            pred_df["true_label_id"] = y[val_idx]
            pred_df["predicted_label_id"] = pred
            pred_df["predicted_label"] = [classes[i] for i in pred]
            pred_df["confidence"] = conf
            prediction_frames.append(pred_df)

            report = classification_report(
                y[val_idx],
                pred,
                labels=np.arange(len(classes)),
                target_names=classes,
                zero_division=0,
            )
            (args.out_dir / f"{model_name}_seed{seed}_classification_report.txt").write_text(report, encoding="utf-8")
            cm = confusion_matrix(y[val_idx], pred, labels=np.arange(len(classes)))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                args.out_dir / f"{model_name}_seed{seed}_confusion_matrix.csv", encoding="utf-8-sig"
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "sherloc_in_situ_model_comparison_by_seed.csv", index=False, encoding="utf-8-sig")
    aggregate = (
        summary.groupby("model")[
            ["accuracy", "macro_f1_all_sherloc_labels", "present_label_macro_f1", "weighted_f1", "internal_tune_macro_f1"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(args.out_dir / "sherloc_in_situ_model_comparison_aggregate.csv", index=False, encoding="utf-8-sig")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(args.out_dir / "sherloc_in_situ_validation_predictions.csv", index=False, encoding="utf-8-sig")
    threshold_all = []
    for model_name, sub in predictions.groupby("model"):
        y_true = sub["true_label_id"].to_numpy(dtype=int)
        y_pred = sub["predicted_label_id"].to_numpy(dtype=int)
        probs = np.zeros((len(sub), len(classes)), dtype=np.float32)
        probs[np.arange(len(sub)), y_pred] = sub["confidence"].to_numpy(dtype=np.float32)
        threshold_all.extend(threshold_rows(y_true, probs, classes, model_name, "sherloc_pooled_random_validation"))
    threshold_df = pd.DataFrame(threshold_all)
    threshold_df.to_csv(args.out_dir / "sherloc_in_situ_confidence_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    threshold_df[threshold_df["threshold"].isin([0.0, 0.5, 0.7, 0.8, 0.9])].to_csv(
        args.out_dir / "sherloc_in_situ_key_thresholds.csv", index=False, encoding="utf-8-sig"
    )

    manifest = {
        "protocol": "Pooled labeled SHERLOC in-situ random-split validation; not independent target/region transfer.",
        "metadata_file": str(args.metadata_file),
        "variant": args.variant,
        "seeds": args.seeds,
        "val_fraction": args.val_fraction,
        "internal_tune_fraction": args.tune_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "classes": classes,
        "label_counts": pooled["model_label"].value_counts().to_dict(),
        "pooled_origin_counts": pooled["pooled_origin"].value_counts().to_dict(),
        "device": str(device),
    }
    (args.out_dir / "sherloc_in_situ_model_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nAggregate:")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
