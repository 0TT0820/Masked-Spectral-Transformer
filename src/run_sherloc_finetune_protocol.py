from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from train_review_comparison import (
    GRID,
    MaskedSpectralTransformer,
    preprocess_spectrum,
)


ROOT = Path(r"D:/dyt/raman/pigeonite")
METADATA_FILE = ROOT / "data" / "metadata_outputs" / "metadata_parent_945.csv"
BASE_RUN = ROOT / "reviewer2_materialized_augmented_results_v1" / "review_ready_20260503_033305"
BASE_MODEL = BASE_RUN / "torch" / "mst.pth"
BASE_MANIFEST = BASE_RUN / "experiment_manifest.json"
OUT_DIR = ROOT / "reviewer2_sherloc_finetune_results"
SEED = 2024


TARGET_MAP = {
    "ss__0186": "Bellegarde",
    "ss__0304": "Dourbes",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def target_name(file_name: str) -> str:
    for prefix, name in TARGET_MAP.items():
        if str(file_name).startswith(prefix):
            return name
    return "Other SHERLOC"


def load_classes() -> list[str]:
    with BASE_MANIFEST.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return list(manifest["classes"])


def load_sherloc_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_FILE)
    mask = df["file_name_clean"].astype(str).str.startswith(tuple(TARGET_MAP))
    df = df[mask].copy()
    df["target"] = df["file_name_clean"].map(target_name)
    df["model_label"] = df["major_category"].astype(str)
    df = df[df["model_label"].isin({"Sulfate", "Phosphate"})].copy()
    df = df.sort_values(["target", "file_name_clean"]).reset_index(drop=True)
    return df


def build_arrays(df: pd.DataFrame, sherloc_min_shift: float = 800.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    masks = []
    for _, row in df.iterrows():
        x, _, mask = preprocess_spectrum(str(row["file_path"]), baseline="poly", smooth=False)
        invalid_low = GRID < sherloc_min_shift
        x[invalid_low, :] = 0.0
        mask[invalid_low] = True
        features.append(x)
        masks.append(mask)
    return np.stack(features).astype(np.float32), np.stack(masks).astype(bool), GRID.astype(np.float32)


class RamanArrayDataset(Dataset):
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


def make_model(classes: list[str], device: torch.device) -> MaskedSpectralTransformer:
    model = MaskedSpectralTransformer(num_classes=len(classes))
    state = torch.load(BASE_MODEL, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def predict(model: nn.Module, ds: Dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    probs = []
    model.eval()
    with torch.no_grad():
        for x, shifts, masks, _ in loader:
            logits = model(x.to(device), shifts.to(device), masks.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def set_finetune_parameters(model: nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if mode == "head_norm":
        modules = [model.norm, model.head]
    elif mode == "head_only":
        modules = [model.head]
    elif mode == "last_block_head":
        modules = [model.encoder.layers[-1], model.norm, model.head]
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def finetune_one_fold(
    base_classes: list[str],
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
) -> tuple[np.ndarray, float]:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = make_model(base_classes, device)
    set_finetune_parameters(model, mode)

    train_ds = RamanArrayDataset(x_train, masks_train, y_train)
    loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best_loss = float("inf")

    for _ in range(epochs):
        model.train()
        for xb, shifts, masks, yb in loader:
            optimizer.zero_grad()
            logits = model(xb.to(device), shifts.to(device), masks.to(device))
            loss = loss_fn(logits, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            best_loss = min(best_loss, float(loss.detach().cpu()))

    test_ds = RamanArrayDataset(x_test, masks_test, y_test)
    return predict(model, test_ds, device), best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit SHERLOC fine-tuning protocol for Reviewer 2 detail comment 1.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", choices=["head_norm", "head_only", "last_block_head", "all"], default="last_block_head")
    parser.add_argument("--sherloc-min-shift", type=float, default=800.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df = load_sherloc_metadata()
    df["label_id"] = df["model_label"].map(class_to_idx)
    if df["label_id"].isna().any():
        missing = sorted(df[df["label_id"].isna()]["model_label"].unique())
        raise RuntimeError(f"SHERLOC labels not in base classes: {missing}")
    y = df["label_id"].to_numpy(dtype=np.int64)
    x, masks, _ = build_arrays(df, sherloc_min_shift=args.sherloc_min_shift)

    log(f"Loaded {len(df)} SHERLOC spectra from Bellegarde/Dourbes")
    log(f"Using base model: {BASE_MODEL}")
    log(f"Device: {device}")

    base_model = make_model(classes, device)
    base_probs = predict(base_model, RamanArrayDataset(x, masks, y), device)
    base_pred = np.argmax(base_probs, axis=1)

    fold_probs = np.zeros_like(base_probs)
    fold_losses = []
    for heldout in range(len(df)):
        train_idx = np.array([i for i in range(len(df)) if i != heldout], dtype=int)
        test_idx = np.array([heldout], dtype=int)
        probs, loss = finetune_one_fold(
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
        )
        fold_probs[test_idx] = probs
        fold_losses.append(loss)
        log(f"Fold {heldout + 1}/{len(df)} held out {df.loc[heldout, 'spectrum_id']}")

    ft_pred = np.argmax(fold_probs, axis=1)
    rows = []
    for i, row in df.iterrows():
        rows.append(
            {
                "spectrum_id": row["spectrum_id"],
                "file_name_clean": row["file_name_clean"],
                "target": row["target"],
                "true_label": row["model_label"],
                "subtype_label": row["subtype_label"],
                "zero_shot_prediction": classes[int(base_pred[i])],
                "zero_shot_confidence": float(base_probs[i, base_pred[i]]),
                "finetuned_loso_prediction": classes[int(ft_pred[i])],
                "finetuned_loso_confidence": float(fold_probs[i, ft_pred[i]]),
                "heldout_fold": i + 1,
            }
        )
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(args.out_dir / "sherloc_loso_predictions.csv", index=False, encoding="utf-8-sig")

    labels_present = sorted(np.unique(y).tolist())
    report_labels = [classes[i] for i in labels_present]
    summary = {
        "protocol": "Earth-domain MST pretraining followed by SHERLOC small-sample fine-tuning; leave-one-SHERLOC-spectrum-out validation.",
        "base_model": str(BASE_MODEL),
        "fine_tuning_samples_per_fold": int(len(df) - 1),
        "held_out_samples_total": int(len(df)),
        "fine_tuning_mode": args.mode,
        "sherloc_valid_range_cm-1": [args.sherloc_min_shift, 4000.0],
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "augmentation_on_sherloc": False,
        "targets": df["target"].value_counts().to_dict(),
        "labels": df["model_label"].value_counts().to_dict(),
        "zero_shot_accuracy": float(accuracy_score(y, base_pred)),
        "zero_shot_macro_f1_all_classes": float(f1_score(y, base_pred, average="macro", zero_division=0)),
        "zero_shot_macro_f1_present_labels": float(f1_score(y, base_pred, labels=labels_present, average="macro", zero_division=0)),
        "finetuned_loso_accuracy": float(accuracy_score(y, ft_pred)),
        "finetuned_loso_macro_f1_all_classes": float(f1_score(y, ft_pred, average="macro", zero_division=0)),
        "finetuned_loso_macro_f1_present_labels": float(f1_score(y, ft_pred, labels=labels_present, average="macro", zero_division=0)),
        "mean_final_training_loss": float(np.mean(fold_losses)),
    }
    (args.out_dir / "sherloc_finetune_protocol_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_text = "Zero-shot report\n" + classification_report(y, base_pred, labels=labels_present, target_names=report_labels, zero_division=0)
    report_text += "\nFine-tuned leave-one-spectrum-out report\n"
    report_text += classification_report(y, ft_pred, labels=labels_present, target_names=report_labels, zero_division=0)
    (args.out_dir / "sherloc_finetune_classification_report.txt").write_text(report_text, encoding="utf-8")

    target_rows = []
    for target, sub in pred_df.groupby("target"):
        idx = sub.index.to_numpy()
        target_rows.append(
            {
                "target": target,
                "n_spectra": int(len(idx)),
                "zero_shot_accuracy": float(accuracy_score(y[idx], base_pred[idx])),
                "finetuned_loso_accuracy": float(accuracy_score(y[idx], ft_pred[idx])),
                "true_labels": "; ".join(f"{k}:{v}" for k, v in sub["true_label"].value_counts().to_dict().items()),
            }
        )
    pd.DataFrame(target_rows).to_csv(args.out_dir / "sherloc_target_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(args.out_dir / "sherloc_finetune_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(args.out_dir / "sherloc_finetune_samples.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
