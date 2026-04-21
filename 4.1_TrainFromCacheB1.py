import os
import glob
import random
from pathlib import Path
import csv  # ✅ 新增

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =========================
# CONFIG（只改这里）
# =========================
CACHE_DIR = Path("dataset/cache")
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2
SEED = 42

NUM_WORKERS = 0  # Windows先0稳定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_OUT = "pointnet_from_cache.pt"

# ✅ 新增：日志 CSV
LOG_CSV = "train_log.csv"


# =========================
# Dataset (load npy only)
# =========================
def collect_ids(cache_dir: Path):
    x_files = sorted(cache_dir.glob("P*_X.npy"))
    ids = []
    for x in x_files:
        pid = x.name.replace("_X.npy", "")
        y = cache_dir / f"{pid}_y.npy"
        if y.exists():
            ids.append(pid)
    return ids


class NpyEarHRTFDataset(Dataset):
    def __init__(self, cache_dir: Path, ids):
        self.cache_dir = cache_dir
        self.ids = ids

        # 读取一个 y 来确定输出维度
        y0 = np.load(self.cache_dir / f"{self.ids[0]}_y.npy")
        self.out_dim = int(y0.reshape(-1).shape[0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        X = np.load(self.cache_dir / f"{pid}_X.npy").astype(np.float32)  # (N,3)
        y = np.load(self.cache_dir / f"{pid}_y.npy").astype(np.float32)  # (D,)
        return torch.from_numpy(X), torch.from_numpy(y), pid


# =========================
# Metric: LSD on logmag targets
# =========================
def lsd_db(pred: torch.Tensor, gt: torch.Tensor) -> float:
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
# Model: PointNet baseline
# =========================
class PointNetEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, emb_dim), nn.ReLU(),
        )

    def forward(self, x):
        feat = self.mlp(x)            # (B,N,emb)
        g, _ = torch.max(feat, dim=1) # (B,emb)
        return g


class PointNetRegressor(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.enc = PointNetEncoder(emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(),
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


def train_one_epoch(model, loader, opt):
    model.train()
    loss_fn = nn.MSELoss()

    mse_sum, lsd_sum, n = 0.0, 0.0, 0
    for X, y, _pid in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        b = X.size(0)
        mse_sum += float(loss.detach().cpu().item()) * b
        lsd_sum += lsd_db(pred, y) * b
        n += b

    return mse_sum / n, lsd_sum / n


@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    loss_fn = nn.MSELoss()

    mse_sum, lsd_sum, n = 0.0, 0.0, 0
    for X, y, _pid in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = loss_fn(pred, y)

        b = X.size(0)
        mse_sum += float(loss.detach().cpu().item()) * b
        lsd_sum += lsd_db(pred, y) * b
        n += b

    return mse_sum / n, lsd_sum / n


def main():
    set_seed(SEED)

    ids = collect_ids(CACHE_DIR)
    if len(ids) < 2:
        raise RuntimeError(f"Not enough cache pairs in {CACHE_DIR}. Need Pxxxx_X.npy and Pxxxx_y.npy")

    print(f"[Cache] Found {len(ids)} subjects")
    print("Device:", DEVICE)

    dataset = NpyEarHRTFDataset(CACHE_DIR, ids)
    out_dim = dataset.out_dim
    print("[Dataset] out_dim =", out_dim)

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * VAL_RATIO)))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    model = PointNetRegressor(out_dim=out_dim, emb_dim=256).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = 1e9

    # ✅ 新增：创建/覆盖 CSV，并写表头
    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "train_lsd", "val_mse", "val_lsd"])

    for epoch in range(1, EPOCHS + 1):
        tr_mse, tr_lsd = train_one_epoch(model, train_loader, opt)
        va_mse, va_lsd = eval_one_epoch(model, val_loader)

        print(f"Epoch {epoch:03d} | train MSE={tr_mse:.4f} LSD={tr_lsd:.3f} | val MSE={va_mse:.4f} LSD={va_lsd:.3f}")

        # ✅ 新增：每个 epoch 追加一行
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_mse, tr_lsd, va_mse, va_lsd])

        if va_lsd < best_val:
            best_val = va_lsd
            torch.save(
                {"model": model.state_dict(), "out_dim": out_dim, "best_val_lsd": best_val, "ids": ids},
                CKPT_OUT
            )

    print("\nSaved best:", CKPT_OUT, "best_val_lsd=", best_val)
    print("Saved log:", LOG_CSV)


if __name__ == "__main__":
    main()
