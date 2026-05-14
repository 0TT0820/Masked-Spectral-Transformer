"""Run target-level SHERLOC transfer experiments.

Protocol:
1. Use the MST pretrained on the 945-spectrum parent dataset with train-time
   physics-aware augmentation.
2. Use only SHERLOC point labels that map unambiguously to the current closed-set
   model classes.
3. Do not augment SHERLOC spectra during fine-tuning.
4. Evaluate by leave-one-target-out: fine-tune on three SHERLOC targets and test
   on the held-out target.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from train_model_comparison import GRID, MaskedSpectralTransformer, preprocess_spectrum


ROOT = Path(r"D:/dyt/raman/pigeonite")
DEFAULT_BASE_RUN = ROOT / "results" / "materialized_augmented_pretraining"
DEFAULT_METADATA = ROOT / "data" / "metadata_outputs" / "metadata_parent_945_plus_sherloc_regions_table1_training_ready.csv"
DEFAULT_OUT = ROOT / "results" / "sherloc_target_transfer"
SEED = 2024


MODEL_LABEL_MAP = {
    "Other Silicate": "Other Silicates",
    "Oxide/Hydroxide": "Oxides/Hydroxides",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_classes(base_run: Path) -> list[str]:
    manifest = json.loads((base_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["classes"])


def load_sherloc_metadata(metadata_file: Path, classes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(metadata_file)
    is_sherloc = df["source_type"].astype(str).str.contains("SHERLOC", case=False, na=False)
    df = df[is_sherloc].copy()
    df = df[df["sherloc_training_label_usable"].astype(str).str.lower().eq("true")].copy()
    df["model_label"] = df["paper_table1_superclass"].replace(MODEL_LABEL_MAP)
    df["base_class_available"] = df["model_label"].isin(classes)
    df["target"] = df["sherloc_target"].astype(str)
    df = df.sort_values(["target", "sherloc_sheet_name", "sherloc_point_name", "model_label"]).reset_index(drop=True)
    return df


def build_arrays(df: pd.DataFrame, sherloc_min_shift: float) -> tuple[np.ndarray, np.ndarray]:
    features = []
    masks = []
    invalid_low = GRID < sherloc_min_shift
    for _, row in df.iterrows():
        x, _, mask = preprocess_spectrum(str(row["file_path"]), baseline="poly", smooth=False)
        x[invalid_low, :] = 0.0
        mask[invalid_low] = True
        features.append(x)
        masks.append(mask)
    return np.stack(features).astype(np.float32), np.stack(masks).astype(bool)


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, masks: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.masks = masks.astype(bool)
        self.y = y.astype(np.int64)
        self.shifts = np.broadcast_to(GRID.reshape(1, -1), (len(self.x), len(GRID))).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.shifts[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def load_model(base_run: Path, classes: list[str], device: torch.device) -> MaskedSpectralTransformer:
    model = MaskedSpectralTransformer(num_classes=len(classes))
    state = torch.load(base_run / "torch" / "mst.pth", map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def predict(model: nn.Module, ds: Dataset, device: torch.device, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for x, shifts, masks, _ in loader:
            logits = model(x.to(device), shifts.to(device), masks.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def set_finetune_parameters(model: nn.Module, mode: str) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    if mode == "head_norm":
        modules = [model.norm, model.head]
    elif mode == "last_block_head":
        modules = [model.encoder.layers[-1], model.norm, model.head]
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(mode)
    params = []
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def finetune(
    base_run: Path,
    classes: list[str],
    x_train: np.ndarray,
    masks_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    masks_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    mode: str,
    l2sp: float,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = load_model(base_run, classes, device)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = set_finetune_parameters(model, mode)
    ds = ArrayDataset(x_train, masks_train, y_train)
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        losses = []
        for xb, shifts, masks, yb in loader:
            opt.zero_grad()
            logits = model(xb.to(device), shifts.to(device), masks.to(device))
            loss = loss_fn(logits, yb.to(device))
            if l2sp > 0:
                reg = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        reg = reg + torch.sum((param - base_state[name].to(device)) ** 2)
                loss = loss + l2sp * reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))

    return predict(model, ArrayDataset(x_test, masks_test, y_test), device), final_loss


def point_any_match_accuracy(df: pd.DataFrame, pred_labels: list[str]) -> float:
    tmp = df.copy()
    tmp["pred"] = pred_labels
    keys = ["target", "sherloc_sheet_name", "sherloc_point_name"]
    correct = []
    for _, sub in tmp.groupby(keys, dropna=False):
        true_set = set(sub["model_label"].astype(str))
        pred_set = set(sub["pred"].astype(str))
        correct.append(bool(true_set & pred_set))
    return float(np.mean(correct)) if correct else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Target-level SHERLOC fine-tuning transfer experiment.")
    parser.add_argument("--base-run", type=Path, default=DEFAULT_BASE_RUN)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--mode", choices=["head_norm", "last_block_head", "all"], default="head_norm")
    parser.add_argument("--l2sp", type=float, default=1e-4)
    parser.add_argument("--sherloc-min-shift", type=float, default=800.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes(args.base_run)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_sherloc = load_sherloc_metadata(args.metadata_file, classes)
    excluded = all_sherloc[~all_sherloc["base_class_available"]].copy()
    df = all_sherloc[all_sherloc["base_class_available"]].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No SHERLOC labels are available in the base model class set.")
    df["label_id"] = df["model_label"].map(class_to_idx).astype(int)
    x, masks = build_arrays(df, args.sherloc_min_shift)
    y = df["label_id"].to_numpy(dtype=np.int64)

    log(f"Using base run: {args.base_run}")
    log(f"Device: {device}")
    log(f"SHERLOC usable rows for current base classes: {len(df)}")
    log(f"Excluded rows whose Table 1 class is absent from 945-class base model: {len(excluded)}")

    base_model = load_model(args.base_run, classes, device)
    zero_probs = predict(base_model, ArrayDataset(x, masks, y), device)
    zero_pred = np.argmax(zero_probs, axis=1)

    ft_probs = np.zeros_like(zero_probs)
    fold_rows = []
    targets = sorted(df["target"].unique())
    for heldout in targets:
        train_idx = np.where(~df["target"].eq(heldout).to_numpy())[0]
        test_idx = np.where(df["target"].eq(heldout).to_numpy())[0]
        probs, loss = finetune(
            args.base_run,
            classes,
            x[train_idx],
            masks[train_idx],
            y[train_idx],
            x[test_idx],
            masks[test_idx],
            y[test_idx],
            device,
            epochs=args.epochs,
            lr=args.lr,
            mode=args.mode,
            l2sp=args.l2sp,
        )
        ft_probs[test_idx] = probs
        fold_rows.append(
            {
                "heldout_target": heldout,
                "train_targets": "; ".join(sorted(set(targets) - {heldout})),
                "n_train_label_rows": int(len(train_idx)),
                "n_test_label_rows": int(len(test_idx)),
                "final_train_loss": loss,
            }
        )
        log(f"Held out {heldout}: train={len(train_idx)}, test={len(test_idx)}")

    ft_pred = np.argmax(ft_probs, axis=1)
    zero_labels = [classes[i] for i in zero_pred]
    ft_labels = [classes[i] for i in ft_pred]
    pred_df = df.copy()
    pred_df["zero_shot_prediction"] = zero_labels
    pred_df["zero_shot_confidence"] = zero_probs[np.arange(len(df)), zero_pred]
    pred_df["finetuned_prediction"] = ft_labels
    pred_df["finetuned_confidence"] = ft_probs[np.arange(len(df)), ft_pred]
    pred_df.to_csv(args.out_dir / "sherloc_target_transfer_predictions.csv", index=False, encoding="utf-8-sig")
    if not excluded.empty:
        excluded.to_csv(args.out_dir / "sherloc_excluded_labels_not_in_base_classes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "sherloc_target_transfer_folds.csv", index=False, encoding="utf-8-sig")

    label_ids_present = sorted(np.unique(y).tolist())
    report_labels = [classes[i] for i in label_ids_present]
    report = "Zero-shot target-pooled report\n"
    report += classification_report(y, zero_pred, labels=label_ids_present, target_names=report_labels, zero_division=0)
    report += "\nFine-tuned leave-one-target-out report\n"
    report += classification_report(y, ft_pred, labels=label_ids_present, target_names=report_labels, zero_division=0)
    (args.out_dir / "sherloc_target_transfer_classification_report.txt").write_text(report, encoding="utf-8")

    target_rows = []
    for target, sub in pred_df.groupby("target"):
        idx = sub.index.to_numpy()
        target_rows.append(
            {
                "target": target,
                "n_label_rows": int(len(sub)),
                "n_unique_points": int(sub.groupby(["sherloc_sheet_name", "sherloc_point_name"]).ngroups),
                "true_labels": "; ".join(f"{k}:{v}" for k, v in sub["model_label"].value_counts().to_dict().items()),
                "zero_shot_accuracy_label_rows": float(accuracy_score(y[idx], zero_pred[idx])),
                "finetuned_accuracy_label_rows": float(accuracy_score(y[idx], ft_pred[idx])),
                "zero_shot_point_any_match_accuracy": point_any_match_accuracy(sub, [zero_labels[i] for i in idx]),
                "finetuned_point_any_match_accuracy": point_any_match_accuracy(sub, [ft_labels[i] for i in idx]),
            }
        )
    target_summary = pd.DataFrame(target_rows)
    target_summary.to_csv(args.out_dir / "sherloc_target_transfer_target_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "protocol": (
            "MST pretrained on 945 parent spectra with train-time physics-aware augmentation; "
            "SHERLOC spectra are not augmented; leave-one-target-out fine-tuning tests transfer "
            "to an unseen SHERLOC target."
        ),
        "base_run": str(args.base_run),
        "metadata_file": str(args.metadata_file),
        "base_classes": classes,
        "n_sherloc_training_usable_rows_in_metadata": int(len(all_sherloc)),
        "n_rows_used_for_current_base_class_set": int(len(df)),
        "n_rows_excluded_class_absent_from_base": int(len(excluded)),
        "excluded_labels": excluded["paper_table1_superclass"].value_counts().to_dict() if not excluded.empty else {},
        "targets": df["target"].value_counts().to_dict(),
        "labels": df["model_label"].value_counts().to_dict(),
        "fine_tune_mode": args.mode,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "l2_sp_regularization": args.l2sp,
        "sherloc_valid_range_cm-1": [args.sherloc_min_shift, 4000.0],
        "zero_shot_accuracy_label_rows": float(accuracy_score(y, zero_pred)),
        "zero_shot_macro_f1_present_labels": float(f1_score(y, zero_pred, labels=label_ids_present, average="macro", zero_division=0)),
        "zero_shot_point_any_match_accuracy": point_any_match_accuracy(df, zero_labels),
        "finetuned_accuracy_label_rows": float(accuracy_score(y, ft_pred)),
        "finetuned_macro_f1_present_labels": float(f1_score(y, ft_pred, labels=label_ids_present, average="macro", zero_division=0)),
        "finetuned_point_any_match_accuracy": point_any_match_accuracy(df, ft_labels),
    }
    (args.out_dir / "sherloc_target_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([summary]).to_csv(args.out_dir / "sherloc_target_transfer_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
