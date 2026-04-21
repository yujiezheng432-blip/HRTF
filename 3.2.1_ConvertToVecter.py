# ==========================================================
# train_pointnet_sonicom_single_dir.py
# ==========================================================
# 目标：
#   用“STL(点云)” -> 预测 “SOFA里某个方向(az,el)的 HRTF log-magnitude (dB)”
#
# 你的数据结构：
#   dataset/
#     P0001.stl
#     P0002.stl
#     ...
#     sofa/
#       P0001_FreeFieldCompMinPhase_48kHz.sofa
#       P0002_FreeFieldCompMinPhase_48kHz.sofa
#       ...
#
# 关键需求：
#   - 只使用 STL 和 SOFA 都存在的 subject 进行训练
#   - 单方向 baseline（先跑通）：默认 az=0, el=0
#
# 运行：
#   python train_pointnet_sonicom_single_dir.py
# ==========================================================

import os
import re
import math
import random
from pathlib import Path

import numpy as np
import h5py
import trimesh

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =========================
# CONFIG（只改这里）
# =========================
DATASET_DIR = Path("dataset")
SOFA_DIR = DATASET_DIR / "sofa"

SOFA_NAME_FMT = "{pid}_FreeFieldCompMinPhase_48kHz.sofa"  # 你给的命名
STL_NAME_FMT = "{pid}.stl"

# 单方向（先跑通）
TARGET_AZ = 0.0
TARGET_EL = 0.0

# 频段（常用）
FMIN_HZ = 200.0
FMAX_HZ = 16000.0

# 点云采样
NUM_POINTS = 1024

# 训练超参
BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2
SEED = 42

# 性能
NUM_WORKERS = 0  # Windows 建议先 0，稳定后再改 2/4
PIN_MEMORY = True

# 保存
OUT_CKPT = "pointnet_single_dir_allsubjects.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Utils: 角度匹配
# =========================
def wrap_angle_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def find_nearest_direction(az_all, el_all, target_az, target_el) -> int:
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    return int(np.argmin(daz**2 + delv**2))


# =========================
# Utils: 点云采样与归一化
# =========================
def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, count=num_points)
    return pts.astype(np.float32)


def normalize_pointcloud(P: np.ndarray) -> np.ndarray:
    P = P.astype(np.float32)
    P = P - P.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(P, axis=1))
    if scale > 0:
        P = P / scale
    return P


# =========================
# Utils: SOFA -> 单方向 log-magnitude(dB)
# =========================
def read_sampling_rate(f: h5py.File) -> float:
    # SONICOM 常见在 Data.SamplingRate；但也给几个兜底
    if "Data.SamplingRate" in f:
        return float(np.array(f["Data.SamplingRate"][:]).squeeze())
    if "Data" in f and "SamplingRate" in f["Data"]:
        return float(np.array(f["Data"]["SamplingRate"][:]).squeeze())
    if "SamplingRate" in f:
        return float(np.array(f["SamplingRate"][:]).squeeze())
    raise RuntimeError("Sampling rate not found in this SOFA file.")


def sofa_single_direction_logmag(
    sofa_path: str,
    target_az: float,
    target_el: float,
    fmin_hz: float,
    fmax_hz: float,
    eps: float = 1e-12,
):
    with h5py.File(sofa_path, "r") as f:
        src = f["SourcePosition"][:]   # (M,3)
        ir = f["Data.IR"][:]           # (M,R,N) typically (M,2,N)
        sr = read_sampling_rate(f)

    az = src[:, 0].astype(np.float32)
    el = src[:, 1].astype(np.float32)
    idx = find_nearest_direction(az, el, target_az, target_el)

    left = ir[idx, 0, :].astype(np.float32)
    right = ir[idx, 1, :].astype(np.float32)

    N = left.shape[0]
    HL = np.fft.rfft(left, n=N)
    HR = np.fft.rfft(right, n=N)

    magL = np.abs(HL)
    magR = np.abs(HR)

    logL = 20.0 * np.log10(magL + eps)
    logR = 20.0 * np.log10(magR + eps)

    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)

    y = np.stack([logL[band], logR[band]], axis=1).astype(np.float32)  # (F,2)
    y = y.reshape(-1).astype(np.float32)  # (F*2,)

    matched = (float(az[idx]), float(el[idx]))
    return y, matched


# =========================
# Metric: LSD (dB) on log-magnitude targets
# =========================
def lsd_db(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # pred, gt: (B, F*2) or (F*2,)
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if gt.ndim == 1:
        gt = gt.unsqueeze(0)

    B, D = pred.shape
    pred2 = pred.view(B, -1, 2)
    gt2 = gt.view(B, -1, 2)

    diff = pred2 - gt2
    per_ear = torch.sqrt(torch.mean(diff * diff, dim=1))  # (B,2)
    return float(torch.mean(per_ear).detach().cpu().item())


# =========================
# Dataset: 只收集 STL+SOFA都存在的 pairs
# =========================
def collect_pairs(dataset_dir: Path, sofa_dir: Path):
    # STL: dataset/Pxxxx.stl
    stl_paths = list(dataset_dir.glob("P*.stl"))

    pairs = []
    for stl_path in stl_paths:
        pid = stl_path.stem  # "P0001"
        sofa_path = sofa_dir / SOFA_NAME_FMT.format(pid=pid)
        if sofa_path.exists():
            pairs.append({"pid": pid, "stl": str(stl_path), "sofa": str(sofa_path)})

    pairs.sort(key=lambda x: x["pid"])
    return pairs


class EarHRTFDataset(Dataset):
    def __init__(self, pairs, num_points, target_az, target_el, fmin_hz, fmax_hz):
        self.pairs = pairs
        self.num_points = num_points
        self.target_az = target_az
        self.target_el = target_el
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz

        # 先用第一个样本确定输出维度（out_dim）
        y0, matched = sofa_single_direction_logmag(
            self.pairs[0]["sofa"], self.target_az, self.target_el, self.fmin_hz, self.fmax_hz
        )
        self.out_dim = int(y0.shape[0])
        self.example_matched = matched

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):


        item = self.pairs[idx]

        print("Loading:", item["pid"])
        mesh = trimesh.load(item["stl"], force="mesh")
        P = sample_points_from_mesh(mesh, self.num_points)
        P = normalize_pointcloud(P)

        y, _ = sofa_single_direction_logmag(
            item["sofa"], self.target_az, self.target_el, self.fmin_hz, self.fmax_hz
        )

        # (N,3), (out_dim,)
        return (
            torch.from_numpy(P).float(),
            torch.from_numpy(y).float(),
            item["pid"],
        )


# =========================
# Model: PointNet (simple, stable baseline)
# =========================
class PointNetEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B,N,3)
        feat = self.mlp(x)            # (B,N,emb)
        g, _ = torch.max(feat, dim=1) # (B,emb)
        return g


class PointNetRegressor(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.enc = PointNetEncoder(emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        g = self.enc(x)
        return self.head(g)


# =========================
# Train / Eval
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, opt, device):
    model.train()
    mse_sum = 0.0
    lsd_sum = 0.0
    n = 0

    loss_fn = nn.MSELoss()

    for P, y, _pid in loader:
        P = P.to(device)  # (B,N,3)
        y = y.to(device)  # (B,D)

        pred = model(P)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        b = P.shape[0]
        mse_sum += float(loss.detach().cpu().item()) * b
        lsd_sum += lsd_db(pred, y) * b
        n += b

    return mse_sum / max(n, 1), lsd_sum / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()

    mse_sum = 0.0
    lsd_sum = 0.0
    n = 0

    for P, y, _pid in loader:
        P = P.to(device)
        y = y.to(device)

        pred = model(P)
        loss = loss_fn(pred, y)

        b = P.shape[0]
        mse_sum += float(loss.detach().cpu().item()) * b
        lsd_sum += lsd_db(pred, y) * b
        n += b

    return mse_sum / max(n, 1), lsd_sum / max(n, 1)


def main():
    set_seed(SEED)

    # 1) 收集 pairs（只保留 stl+sofa 都存在的）
    pairs = collect_pairs(DATASET_DIR, SOFA_DIR)
    if len(pairs) < 2:
        raise RuntimeError(
            f"Not enough valid pairs. Found {len(pairs)} pairs under:\n"
            f"  STL:  {DATASET_DIR}\n"
            f"  SOFA: {SOFA_DIR}\n"
            f"Expected: dataset/Pxxxx.stl and dataset/sofa/Pxxxx_FreeFieldCompMinPhase_48kHz.sofa"
        )

    print(f"[Pairs] Using {len(pairs)} subjects with BOTH STL and SOFA.")
    print("[Example]", pairs[0]["pid"])
    print("Device:", DEVICE)

    # 2) Dataset / split
    dataset = EarHRTFDataset(
        pairs=pairs,
        num_points=NUM_POINTS,
        target_az=TARGET_AZ,
        target_el=TARGET_EL,
        fmin_hz=FMIN_HZ,
        fmax_hz=FMAX_HZ,
    )
    out_dim = dataset.out_dim
    print(f"[Dataset] Output dim = {out_dim}")
    print(f"[Dataset] Example matched az/el in file = {dataset.example_matched}")

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * VAL_RATIO)))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    # 3) Model
    model = PointNetRegressor(out_dim=out_dim, emb_dim=256).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")

    # 4) Train loop
    for epoch in range(1, EPOCHS + 1):
        tr_mse, tr_lsd = train_one_epoch(model, train_loader, opt, DEVICE)
        va_mse, va_lsd = eval_one_epoch(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:03d} | "
            f"train MSE={tr_mse:.4f} LSD={tr_lsd:.3f} dB | "
            f"val MSE={va_mse:.4f} LSD={va_lsd:.3f} dB"
        )

        # 保存最好模型（按 val LSD）
        if va_lsd < best_val:
            best_val = va_lsd
            torch.save(
                {
                    "model": model.state_dict(),
                    "out_dim": out_dim,
                    "num_points": NUM_POINTS,
                    "target_az": TARGET_AZ,
                    "target_el": TARGET_EL,
                    "fmin_hz": FMIN_HZ,
                    "fmax_hz": FMAX_HZ,
                    "best_val_lsd": best_val,
                    "pairs_used": len(pairs),
                },
                OUT_CKPT,
            )

    print(f"\nSaved best checkpoint: {OUT_CKPT} (best val LSD={best_val:.3f} dB)")


if __name__ == "__main__":
    main()
