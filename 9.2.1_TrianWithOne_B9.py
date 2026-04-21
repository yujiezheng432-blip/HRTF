from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


PID_RE = re.compile(r"(P\d{4})")

# 固定种子去除随机性
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_pid(path: Path) -> str:
    m = PID_RE.search(path.stem)
    if not m:
        raise ValueError(f"Cannot extract subject id from: {path.name}")
    return m.group(1)

#支持变化长度点云输入
def collate_variable_points(batch):
    pids, xs, ys = zip(*batch)
    return list(pids), list(xs), torch.stack(ys, dim=0)

#读取HRTF区域，将复数转化成dB数据
def restore_all_direction_hrtf(pid: str, hrtf_dir: Path, eps: float = 1e-8) -> tuple[np.ndarray, dict]:
    hrtf_path = hrtf_dir / f"{pid}_AllDirectionHRTF.npy"
    meta_path = hrtf_dir / f"{pid}_DirBins.npy"

    if not hrtf_path.exists():
        raise FileNotFoundError(f"HRTF file not found: {hrtf_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"DirBins/meta file not found: {meta_path}")

    y_raw = np.load(hrtf_path, allow_pickle=True)
    meta = np.load(meta_path, allow_pickle=True).item()

    if "hrtf_shape_sorted" not in meta:
        raise KeyError(f"{meta_path.name} does not contain 'hrtf_shape_sorted'")

    M, R, F = map(int, meta["hrtf_shape_sorted"])
    if R != 2:
        raise ValueError(f"{pid}: expected ear dim R=2, got {R}")

    if y_raw.ndim == 3:
        if y_raw.shape != (M, R, F):
            raise ValueError(f"{pid}: HRTF shape mismatch, got {y_raw.shape}, expected {(M, R, F)}")
        y = y_raw
    elif y_raw.ndim == 1:
        expected_len = 2 * M * F
        if y_raw.shape[0] != expected_len:
            raise ValueError(f"{pid}: flattened HRTF length mismatch, got {y_raw.shape[0]}, expected {expected_len}")
        left = y_raw[: M * F].reshape(M, F)
        right = y_raw[M * F :].reshape(M, F)
        y = np.stack([left, right], axis=1)
    else:
        raise ValueError(f"{pid}: unsupported HRTF shape {y_raw.shape}")

    if np.iscomplexobj(y):
        y = 20.0 * np.log10(np.maximum(np.abs(y), eps)).astype(np.float32)
    else:
        y = y.astype(np.float32)

    return y, meta

#拼接路径，然后将xxx列入队列等待
def resolve_point_path(pid: str, point_dir: Path) -> Path:
    path = point_dir / f"{pid}_HeadNPY.npy"
    if path.exists():
        return path
    candidates = sorted(point_dir.glob(f"{pid}*.npy"))
    if not candidates:
        raise FileNotFoundError(f"Point cloud npy for {pid} not found in {point_dir}")
    return candidates[0]

#加载PCA训练
class VariablePointPCADataset(Dataset):
    def __init__(self, pids: Sequence[str], point_dir: Path, coeff_map: Dict[str, np.ndarray]):
        self.pids = list(pids)
        self.point_dir = point_dir
        self.coeff_map = coeff_map
        self.cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if len(self.pids) == 0:
            raise ValueError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.pids)

    def _load_pair(self, pid: str) -> tuple[torch.Tensor, torch.Tensor]:
        if pid in self.cache:
            return self.cache[pid]

        x = np.load(resolve_point_path(pid, self.point_dir)).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"{pid}: point cloud must have shape (N,3), got {x.shape}")

        y = self.coeff_map[pid].astype(np.float32)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        self.cache[pid] = (x_t, y_t)
        return x_t, y_t

    def __getitem__(self, idx: int):
        pid = self.pids[idx]
        x_t, y_t = self._load_pair(pid)
        return pid, x_t, y_t

#神经网络部分
class BasicPointNetPCARegressor(nn.Module):
    def __init__(self, coeff_dim: int):
        super().__init__()
        self.coeff_dim = coeff_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, coeff_dim),
        )

    def encode_one(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.point_mlp(x)
        global_feat = feat.max(dim=0).values
        return global_feat

    def forward(self, batch_points: List[torch.Tensor]) -> torch.Tensor:
        global_feats = [self.encode_one(x) for x in batch_points]
        global_feats = torch.stack(global_feats, dim=0)
        return self.head(global_feats)


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def mae_metric_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse_metric_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def lsd_metric_np(pred: np.ndarray, target: np.ndarray) -> float:
    diff2 = (pred - target) ** 2
    lsd_each = np.sqrt(np.mean(diff2, axis=-1) + 1e-12)
    return float(np.mean(lsd_each))


def reconstruct_from_coeff_numpy(coeff: np.ndarray, pca: PCA, out_shape: tuple[int, int, int]) -> np.ndarray:
    flat = pca.inverse_transform(coeff)
    if flat.ndim == 1:
        flat = flat[None, :]
    return flat.reshape(flat.shape[0], *out_shape).astype(np.float32)


def run_one_epoch_train(model, loader, device, optimizer, criterion):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mse = 0.0
    total_count = 0

    for _, xs, y in loader:
        xs = [x.to(device) for x in xs]
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
            pred = model(xs)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bsz = y.size(0)
        total_count += bsz
        total_loss += loss.item() * bsz
        total_mse += mse_metric(pred, y) * bsz

    return {"loss": total_loss / total_count, "coeff_mse": total_mse / total_count}


@torch.no_grad()
def evaluate_reconstruction(model, loader, device, criterion, pca, flat_gt_map, out_shape):
    model.eval()
    total_coeff_loss = 0.0
    total_count = 0
    all_pred = []
    all_true = []

    for pids, xs, y in loader:
        xs = [x.to(device) for x in xs]
        y = y.to(device)
        pred_coeff = model(xs)
        loss = criterion(pred_coeff, y)

        pred_coeff_np = pred_coeff.detach().cpu().numpy()
        pred_hrtf = reconstruct_from_coeff_numpy(pred_coeff_np, pca, out_shape)

        true_hrtf = np.stack([flat_gt_map[pid].reshape(out_shape).astype(np.float32) for pid in pids], axis=0)

        bsz = len(pids)
        total_count += bsz
        total_coeff_loss += loss.item() * bsz
        all_pred.append(pred_hrtf)
        all_true.append(true_hrtf)

    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    return {
        "coeff_mse": total_coeff_loss / total_count,
        "mse": float(np.mean((all_pred - all_true) ** 2)),
        "mae": mae_metric_np(all_pred, all_true),
        "rmse": rmse_metric_np(all_pred, all_true),
        "lsd": lsd_metric_np(all_pred, all_true),
    }


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, current: float) -> bool:
        if current < self.best - self.min_delta:
            self.best = current
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def write_report(report_path, all_pids, train_pids, val_pids, test_pids, point_dir, out_shape, coeff_dim, metrics, args):
    point_sizes = {}
    for pid in all_pids:
        p = resolve_point_path(pid, point_dir)
        x = np.load(p, mmap_mode="r")
        point_sizes[pid] = tuple(x.shape)

    M, R, F = out_shape
    lines = []
    lines.append("Basic9 PCA Report")
    lines.append("=" * 60)
    lines.append(f"Total subjects: {len(all_pids)}")
    lines.append(f"Train / Val / Test: {len(train_pids)} / {len(val_pids)} / {len(test_pids)}")
    lines.append(f"Original output shape (M,2,F): {out_shape}")
    lines.append(f"Original flattened output dim: {M * R * F}")
    lines.append(f"PCA coefficient dim: {coeff_dim}")
    lines.append("")
    lines.append("Point-cloud sizes (per subject)")
    lines.append("-" * 60)
    for pid in all_pids:
        lines.append(f"{pid}: {point_sizes[pid]}")
    lines.append("")
    lines.append("Model dimensions")
    lines.append("-" * 60)
    lines.append("Input point shape: (N_i, 3)  [variable-length]")
    lines.append("Point MLP: 3 -> 64 -> 128 -> 256")
    lines.append("Global max pooling: (N_i, 256) -> (256,)")
    lines.append("Regression head: 256 -> 512 -> 512 -> PCA_coeff")
    lines.append(f"PCA coeff output dim: {coeff_dim}")
    lines.append(f"Reconstructed output dim: ({M}, {R}, {F})")
    lines.append("")
    lines.append("Training setup")
    lines.append("-" * 60)
    lines.append(f"Epochs: {args.epochs}")
    lines.append(f"Batch size: {args.batch_size}")
    lines.append(f"Learning rate: {args.lr}")
    lines.append(f"Weight decay: {args.weight_decay}")
    lines.append(f"Early stopping patience: {args.patience}")
    lines.append(f"Early stopping min_delta: {args.min_delta}")
    lines.append(f"PCA requested dim: {args.pca_dim}")
    lines.append("Training loss: PCA coefficient MSE")
    lines.append("Validation/Test metrics: reconstructed LSD, MAE, RMSE, MSE")
    lines.append("")
    lines.append("Best / Final metrics")
    lines.append("-" * 60)
    for k, v in metrics.items():
        lines.append(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_flat_label_maps(pids, hrtf_dir):
    flat_map = {}
    out_shape = None
    ref_meta = None
    for pid in pids:
        y, meta = restore_all_direction_hrtf(pid, hrtf_dir)
        if out_shape is None:
            out_shape = tuple(y.shape)
            ref_meta = meta
        elif tuple(y.shape) != out_shape:
            raise ValueError(f"{pid}: output shape mismatch {y.shape} vs {out_shape}")
        flat_map[pid] = y.reshape(-1).astype(np.float32)
    return flat_map, out_shape, ref_meta


def main():
    parser = argparse.ArgumentParser(description="Basic9 PCA training script")
    parser.add_argument("--point_dir", type=Path, default=Path("dataset/HeadNPY"))
    parser.add_argument("--hrtf_dir", type=Path, default=Path("dataset/HRTF_AllDirection"))
    parser.add_argument("--out_dir", type=Path, default=Path("Basic9"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    point_files = sorted(args.point_dir.glob("P*_HeadNPY.npy"))
    if not point_files:
        point_files = sorted(args.point_dir.glob("P*.npy"))
    if not point_files:
        raise FileNotFoundError(f"No point cloud npy files found in {args.point_dir}")

    point_pids = [extract_pid(p) for p in point_files]
    hrtf_pids = {extract_pid(p) for p in args.hrtf_dir.glob("P*_AllDirectionHRTF.npy")}
    meta_pids = {extract_pid(p) for p in args.hrtf_dir.glob("P*_DirBins.npy")}
    all_pids = sorted(set(pid for pid in point_pids if pid in hrtf_pids and pid in meta_pids))

    if len(all_pids) < 30:
        raise ValueError(f"Need at least 30 matched subjects for split, got {len(all_pids)}")

    test_pids = all_pids[-20:]
    remain_pids = all_pids[:-20]
    val_count = max(20, int(round(len(remain_pids) * 0.1)))
    val_pids = remain_pids[-val_count:]
    train_pids = remain_pids[:-val_count]

    if len(train_pids) == 0:
        raise ValueError("Train set is empty after split.")

    print(f"Matched subjects: {len(all_pids)}")
    print(f"Train / Val / Test = {len(train_pids)} / {len(val_pids)} / {len(test_pids)}")

    flat_gt_map, out_shape, ref_meta = build_flat_label_maps(all_pids, args.hrtf_dir)
    out_dim = int(np.prod(out_shape))

    train_matrix = np.stack([flat_gt_map[pid] for pid in train_pids], axis=0)
    max_valid_pca = min(len(train_pids), out_dim)
    coeff_dim = min(args.pca_dim, max_valid_pca)
    if coeff_dim < 2:
        raise ValueError(f"Effective PCA dimension too small: {coeff_dim}")

    print(f"Original output shape: {out_shape}")
    print(f"Original flattened dim: {out_dim}")
    print(f"Using PCA coeff dim: {coeff_dim}")

    pca = PCA(n_components=coeff_dim, svd_solver="randomized", random_state=args.seed)
    train_coeff = pca.fit_transform(train_matrix).astype(np.float32)

    coeff_map = {}
    for pid, coeff in zip(train_pids, train_coeff):
        coeff_map[pid] = coeff

    for split_pids in [val_pids, test_pids]:
        mat = np.stack([flat_gt_map[pid] for pid in split_pids], axis=0)
        coeffs = pca.transform(mat).astype(np.float32)
        for pid, coeff in zip(split_pids, coeffs):
            coeff_map[pid] = coeff

    train_ds = VariablePointPCADataset(train_pids, args.point_dir, coeff_map)
    val_ds = VariablePointPCADataset(val_pids, args.point_dir, coeff_map)
    test_ds = VariablePointPCADataset(test_pids, args.point_dir, coeff_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_variable_points)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_variable_points)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_variable_points)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicPointNetPCARegressor(coeff_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopper(args.patience, args.min_delta)

    best_model_path = args.out_dir / "best_model_pca.pth"
    best_metrics_path = args.out_dir / "best_metrics_pca.txt"
    report_path = args.out_dir / "Report_9.txt"

    best_epoch = -1
    best_val_lsd = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_train(model, train_loader, device, optimizer, criterion)
        val_metrics = evaluate_reconstruction(model, val_loader, device, criterion, pca, flat_gt_map, out_shape)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        print(f"Epoch {epoch:03d} | train coeffMSE {train_metrics['coeff_mse']:.6f} | val LSD {val_metrics['lsd']:.6f} MSE {val_metrics['mse']:.6f}")

        current_val_lsd = val_metrics["lsd"]
        if best_val_lsd is None or current_val_lsd < best_val_lsd:
            best_val_lsd = current_val_lsd
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "coeff_dim": coeff_dim,
                "out_shape": out_shape,
                "pca_components": pca.components_.astype(np.float32),
                "pca_mean": pca.mean_.astype(np.float32),
                "pca_explained_variance": pca.explained_variance_.astype(np.float32),
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
                "best_epoch": best_epoch,
                "best_val_lsd": best_val_lsd,
                "train_pids": train_pids,
                "val_pids": val_pids,
                "test_pids": test_pids,
                "point_dir": str(args.point_dir),
                "hrtf_dir": str(args.hrtf_dir),
                "frequency_bins_hz": ref_meta.get("frequency_bins_hz", None),
                "source_position_sorted_deg_m": ref_meta.get("source_position_sorted_deg_m", None),
            }
            torch.save(checkpoint, best_model_path)

        if early_stopper.step(current_val_lsd):
            print(f"Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_val_metrics = evaluate_reconstruction(model, val_loader, device, criterion, pca, flat_gt_map, out_shape)
    test_metrics = evaluate_reconstruction(model, test_loader, device, criterion, pca, flat_gt_map, out_shape)

    final_metrics = {
        "best_epoch": best_epoch,
        "coeff_dim": coeff_dim,
        "best_val_coeff_mse": best_val_metrics["coeff_mse"],
        "best_val_mse": best_val_metrics["mse"],
        "best_val_lsd": best_val_metrics["lsd"],
        "best_val_mae": best_val_metrics["mae"],
        "best_val_rmse": best_val_metrics["rmse"],
        "test_coeff_mse": test_metrics["coeff_mse"],
        "test_mse": test_metrics["mse"],
        "test_lsd": test_metrics["lsd"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "device": str(device),
    }

    with open(best_metrics_path, "w", encoding="utf-8") as f:
        for k, v in final_metrics.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
        f.write("\nHistory (JSON lines)\n")
        for row in history:
            f.write(json.dumps(row) + "\n")

    write_report(report_path, all_pids, train_pids, val_pids, test_pids, args.point_dir, out_shape, coeff_dim, final_metrics, args)

    print("\nTraining finished.")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best metrics saved to: {best_metrics_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
