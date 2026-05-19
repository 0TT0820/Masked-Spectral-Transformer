"""Pooled random-split validation on all labeled SHERLOC DUV spectra.

This protocol combines the labeled SHERLOC fine-tuning pool and the previous
external-validation rows into one supervised SHERLOC adaptation dataset. It then
uses random repeated train/validation splits to estimate point-level adaptation
performance.

Important: this is not an independent target/region transfer test. Adjacent
points from the same target or scan may occur in both train and validation
sets, so the result should be reported as pooled SHERLOC random-split
validation rather than external validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from train_model_comparison import RamanDataset, class_weights, predict_torch
from run_review_updated_training_v2 import (
    METADATA_V2,
    SEED,
    load_partial_mst_state,
    load_v2_metadata,
    make_mst_from_checkpoint_config,
    sherloc_split,
)
from run_sherloc_preprocessing_trials_v2 import build_sherloc_arrays


def metrics(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "present_label_macro_f1": float(
            f1_score(y_true, pred, labels=np.unique(y_true), average="macro", zero_division=0)
        ),
        "predicted_labels": ";".join(sorted({labels[i] for i in np.unique(pred)})),
    }


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
        return train_idx, val_idx, "random_no_stratification"
    train_norm, val_idx = train_test_split(
        normal_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=y[normal_idx],
    )
    train_idx = np.concatenate([train_norm, singleton_idx])
    return np.sort(train_idx), np.sort(val_idx), "stratified_for_labels_with_n>=2_singletons_forced_to_train"


def set_trainable(model: nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if mode == "norm_head":
        modules = [model.norm, model.head]
    elif mode == "last_block_norm_head":
        modules = [model.encoder.layers[-1], model.norm, model.head]
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def train_one_split(
    base_checkpoint: Path,
    base_classes: list[str],
    all_classes: list[str],
    x: np.ndarray,
    masks: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mode: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    split_name: str,
) -> tuple[dict, np.ndarray]:
    model, state, config = make_mst_from_checkpoint_config(base_checkpoint, len(all_classes))
    load_partial_mst_state(model, state, base_classes, all_classes)
    set_trainable(model, mode)
    model.to(device)

    train_loader = DataLoader(RamanDataset(x[train_idx], masks[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RamanDataset(x[val_idx], masks[val_idx], y[val_idx]), batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], len(all_classes)).to(device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xx, shifts, mask, yy in train_loader:
            xx, shifts, mask, yy = xx.to(device), shifts.to(device), mask.to(device), yy.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xx, shifts, mask), yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probs, yy_true = predict_torch(model, val_loader, device)
        val_macro = float(f1_score(yy_true, np.argmax(probs, axis=1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_macro})
        if val_macro > best_val:
            best_val = val_macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    probs, yy_true = predict_torch(model, val_loader, device)
    result = {
        "split": split_name,
        "mode": mode,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "best_val_macro_f1_during_training": best_val,
        "base_config": json.dumps(config),
        **metrics(yy_true, probs, all_classes),
    }
    pd.DataFrame(history).to_csv(out_dir / f"{split_name}_{mode}_history.csv", index=False, encoding="utf-8-sig")
    report = classification_report(
        yy_true,
        np.argmax(probs, axis=1),
        labels=np.arange(len(all_classes)),
        target_names=all_classes,
        zero_division=0,
    )
    (out_dir / f"{split_name}_{mode}_classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(yy_true, np.argmax(probs, axis=1), labels=np.arange(len(all_classes)))
    pd.DataFrame(cm, index=all_classes, columns=all_classes).to_csv(
        out_dir / f"{split_name}_{mode}_confusion_matrix.csv", encoding="utf-8-sig"
    )
    return result, probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled random validation on all labeled SHERLOC spectra.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, default=METADATA_V2)
    parser.add_argument("--variant", default="despike_sg11_asls")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--modes", nargs="+", default=["norm_head", "last_block_norm_head"])
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.run_dir / "sherloc_pooled_random_validation"
    cache_dir = args.run_dir / "_sherloc_preprocessing_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_summary = json.loads((args.run_dir / "sherloc_finetune" / "sherloc_finetune_v2_summary.json").read_text(encoding="utf-8"))
    base_checkpoint = Path(ft_summary["base_checkpoint"])
    base_classes = ft_summary["base_classes"]
    all_classes = ft_summary["all_classes_after_sherloc"]
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    pooled = make_pooled_sherloc(args.metadata_file)
    pooled.to_csv(out_dir / "pooled_sherloc_labeled_samples.csv", index=False, encoding="utf-8-sig")
    x, masks, stats = build_sherloc_arrays(pooled, cache_dir, "sherloc_pooled_labeled", args.variant, args.refresh_cache)
    stats.to_csv(out_dir / "pooled_sherloc_preprocessing_stats.csv", index=False, encoding="utf-8-sig")
    y = pooled["model_label"].map(class_to_idx).to_numpy(dtype=np.int64)

    rows = []
    for seed in args.seeds:
        train_idx, val_idx, split_note = split_indices(y, seed, args.val_fraction)
        split_df = pooled.copy()
        split_df["pooled_random_split"] = "unused"
        split_df.loc[train_idx, "pooled_random_split"] = "train"
        split_df.loc[val_idx, "pooled_random_split"] = "validation"
        split_df.to_csv(out_dir / f"split_seed_{seed}.csv", index=False, encoding="utf-8-sig")
        for mode in args.modes:
            split_name = f"seed{seed}"
            print(f"Training pooled SHERLOC split {split_name}, mode={mode}", flush=True)
            result, _ = train_one_split(
                base_checkpoint,
                base_classes,
                all_classes,
                x,
                masks,
                y,
                train_idx,
                val_idx,
                mode,
                args.epochs,
                args.lr,
                args.batch_size,
                device,
                out_dir,
                split_name,
            )
            result["seed"] = seed
            result["split_note"] = split_note
            rows.append(result)
            pd.DataFrame(rows).to_csv(out_dir / "pooled_random_validation_summary.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(rows)
    aggregate = (
        summary.groupby("mode")[["accuracy", "macro_f1", "weighted_f1", "present_label_macro_f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(out_dir / "pooled_random_validation_aggregate.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "protocol": "All labeled SHERLOC rows pooled; repeated random validation split. Not independent target transfer.",
        "variant": args.variant,
        "seeds": args.seeds,
        "val_fraction": args.val_fraction,
        "modes": args.modes,
        "label_counts": pooled["model_label"].value_counts().to_dict(),
        "pooled_origin_counts": pooled["pooled_origin"].value_counts().to_dict(),
        "singleton_labels_forced_to_train": [k for k, v in pooled["model_label"].value_counts().items() if v < 2],
        "device": str(device),
    }
    (out_dir / "pooled_random_validation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nAggregate:")
    print(aggregate.to_string(index=False))
    print(f"Saved pooled SHERLOC random-validation results to: {out_dir}")


if __name__ == "__main__":
    main()
