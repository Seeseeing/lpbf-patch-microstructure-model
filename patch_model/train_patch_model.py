import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "P",
    "v",
    "h",
    "x_norm",
    "y_norm",
    "signed_distance_to_boundary",
    "T_peak",
    "cooling_rate",
]

CLASS_NAMES = ["coarse", "fine", "boundary"]


class PatchMultiTaskMLP(nn.Module):
    """Shared MLP backbone with two task heads.

    Input:
        8 normalized features.

    Outputs:
        class_logits: three values for coarse/fine/boundary classification.
        grain_size: one normalized regression value.
    """

    def __init__(self, input_dim=8, hidden1=32, hidden2=16, dropout=0.10):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),
        )

        self.classification_head = nn.Linear(hidden2, 3)
        self.regression_head = nn.Linear(hidden2, 1)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classification_head(features)
        grain_size = self.regression_head(features).squeeze(1)
        return class_logits, grain_size


def read_patch_csv(path):
    """Read the patch-level CSV and return arrays."""
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip blank template rows.
            if not row.get("run_id"):
                continue
            rows.append(row)

    if not rows:
        raise ValueError("No data rows found. Fill the CSV template or use synthetic data.")

    run_id = np.array([int(float(row["run_id"])) for row in rows])
    patch_id = np.array([row["patch_id"] for row in rows])

    x = np.array(
        [[float(row[col]) for col in FEATURE_COLUMNS] for row in rows],
        dtype=np.float32,
    )
    zone = np.array([int(float(row["zone_label"])) for row in rows], dtype=np.int64)
    grain = np.array([float(row["local_grain_size"]) for row in rows], dtype=np.float32)

    if np.any((zone < 0) | (zone > 2)):
        raise ValueError("zone_label must be 0, 1, or 2.")

    return {
        "run_id": run_id,
        "patch_id": patch_id,
        "x": x,
        "zone": zone,
        "grain": grain,
    }


def fit_standardizer(values):
    """Fit mean/std normalization on training data only."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standard(values, mean, std):
    return ((values - mean) / std).astype(np.float32)


def fit_target_scaler(values):
    """Scale grain size to zero mean and unit variance for stable training."""
    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0:
        std = 1.0
    return mean, std


def inverse_target(values, mean, std):
    return values * std + mean


def classification_metrics(y_true, y_pred):
    """Compute accuracy, macro-F1, and a 3x3 confusion matrix."""
    confusion = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1

    accuracy = float((y_true == y_pred).mean())

    f1_scores = []
    for cls in range(3):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": confusion.tolist(),
    }


def regression_metrics(y_true, y_pred):
    """Compute MAE, RMSE, and R2 in original grain-size units."""
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_one_split(data, train_runs, val_runs, args, split_name):
    """Train on full runs and validate on full held-out runs."""
    train_mask = np.isin(data["run_id"], train_runs)
    val_mask = np.isin(data["run_id"], val_runs)

    x_train_raw = data["x"][train_mask]
    x_val_raw = data["x"][val_mask]
    zone_train = data["zone"][train_mask]
    zone_val = data["zone"][val_mask]
    grain_train_raw = data["grain"][train_mask]
    grain_val_raw = data["grain"][val_mask]

    x_mean, x_std = fit_standardizer(x_train_raw)
    grain_mean, grain_std = fit_target_scaler(grain_train_raw)

    x_train = transform_standard(x_train_raw, x_mean, x_std)
    x_val = transform_standard(x_val_raw, x_mean, x_std)
    grain_train = ((grain_train_raw - grain_mean) / grain_std).astype(np.float32)
    grain_val = ((grain_val_raw - grain_mean) / grain_std).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(zone_train),
        torch.from_numpy(grain_train),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > args.batch_size,
    )

    model = PatchMultiTaskMLP(dropout=args.dropout).to(args.device)
    class_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = math.inf
    best_state = None

    x_val_t = torch.from_numpy(x_val).to(args.device)
    zone_val_t = torch.from_numpy(zone_val).to(args.device)
    grain_val_t = torch.from_numpy(grain_val).to(args.device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, zone_b, grain_b in train_loader:
            xb = xb.to(args.device)
            zone_b = zone_b.to(args.device)
            grain_b = grain_b.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            logits, grain_pred = model(xb)

            class_loss = class_loss_fn(logits, zone_b)
            reg_loss = reg_loss_fn(grain_pred, grain_b)
            loss = class_loss + args.regression_weight * reg_loss

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            train_count += len(xb)

        model.eval()
        with torch.no_grad():
            val_logits, val_grain_pred = model(x_val_t)
            val_class_loss = class_loss_fn(val_logits, zone_val_t)
            val_reg_loss = reg_loss_fn(val_grain_pred, grain_val_t)
            val_loss = val_class_loss + args.regression_weight * val_reg_loss
            val_loss_value = float(val_loss.item())

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_loss = train_loss_sum / max(train_count, 1)
            print(
                f"{split_name} epoch={epoch:04d} "
                f"train_loss={train_loss:.5f} val_loss={val_loss_value:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, grain_pred_norm = model(x_val_t)
        zone_pred = logits.argmax(dim=1).cpu().numpy()
        grain_pred_norm = grain_pred_norm.cpu().numpy()

    grain_pred = inverse_target(grain_pred_norm, grain_mean, grain_std)

    cls_metrics = classification_metrics(zone_val, zone_pred)
    reg_metrics = regression_metrics(grain_val_raw, grain_pred)

    split_dir = args.output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "class_names": CLASS_NAMES,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "grain_mean": grain_mean,
        "grain_std": grain_std,
        "train_runs": [int(x) for x in train_runs],
        "val_runs": [int(x) for x in val_runs],
    }
    torch.save(checkpoint, split_dir / "model.pt")

    metrics = {
        "split_name": split_name,
        "train_runs": [int(x) for x in train_runs],
        "val_runs": [int(x) for x in val_runs],
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "best_val_loss": best_val_loss,
        **cls_metrics,
        **reg_metrics,
    }

    with (split_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def write_metrics_csv(path, rows):
    """Write one compact metrics table."""
    fields = [
        "split_name",
        "train_runs",
        "val_runs",
        "n_train",
        "n_val",
        "accuracy",
        "macro_f1",
        "mae",
        "rmse",
        "r2",
        "best_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            compact = {field: row.get(field) for field in fields}
            compact["train_runs"] = " ".join(map(str, compact["train_runs"]))
            compact["val_runs"] = " ".join(map(str, compact["val_runs"]))
            writer.writerow(compact)


def summarize_metric_rows(rows):
    keys = ["accuracy", "macro_f1", "mae", "rmse", "r2"]
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("patch_model_runs"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--regression-weight",
        type=float,
        default=1.0,
        help="Weight for the grain-size regression loss.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = read_patch_csv(args.data)
    unique_runs = np.array(sorted(np.unique(data["run_id"])))

    print(f"Loaded {len(data['run_id'])} patches from {len(unique_runs)} runs")
    print(f"Runs: {unique_runs.tolist()}")
    print(f"Device: {args.device}")

    # Primary validation: Leave-One-Run-Out CV.
    loo_rows = []
    for held_out_run in unique_runs:
        train_runs = unique_runs[unique_runs != held_out_run]
        val_runs = np.array([held_out_run])
        split_name = f"fold_leave_run_{int(held_out_run)}"
        metrics = train_one_split(data, train_runs, val_runs, args, split_name)
        loo_rows.append(metrics)

    write_metrics_csv(args.output_dir / "metrics_leave_one_run_out.csv", loo_rows)

    # Secondary demonstration split:
    # train = Run 2, 13, 14, 15, 16; test = Run 17, 18.
    fixed_rows = []
    fixed_train = np.array([2, 13, 14, 15, 16])
    fixed_val = np.array([17, 18])
    if set(fixed_train).issubset(set(unique_runs)) and set(fixed_val).issubset(set(unique_runs)):
        fixed_metrics = train_one_split(
            data,
            fixed_train,
            fixed_val,
            args,
            "fixed_train_2_13_14_15_16_test_17_18",
        )
        fixed_rows.append(fixed_metrics)
        write_metrics_csv(args.output_dir / "metrics_fixed_split.csv", fixed_rows)

    summary = {
        "leave_one_run_out": summarize_metric_rows(loo_rows),
        "fixed_split": fixed_rows[0] if fixed_rows else None,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
