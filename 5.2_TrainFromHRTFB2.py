import random
from pathlib import Path
import csv

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# =========================
# CONFIG（只改这里）
# =========================
CACHE_DIR = Path("dataset/cache")                 # X: Pxxxx_X.npy
HRTF_DIR  = Path("dataset/HRTF_UpperBins")        # H: Pxxxx_HRTF_512bins.npy (complex)
SOFA_DIR  = Path("dataset/sofa")                  # 用于 SourcePosition 找(0,0)方向

HRTF_SUFFIX = "_HRTF_512bins.npy"                 # 按你的保存命名
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

# TARGET_AZ = 0.0
TARGET_AZ = 270.0

TARGET_EL = 0.0
W_EL = 1.0

BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2
SEED = 42

NUM_WORKERS = 0  # Windows先0稳定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_OUT = "pointnet_front_512bins.pt"
LOG_CSV = "train_log_front_512bins.csv"

# Early Stopping
PATIENCE = 10
MIN_DELTA = 1e-4

# 数值稳定
EPS = 1e-12

# 训练结束后随机画图数量
N_PLOT_SUBJECTS = 3


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx


def get_front_index_from_sofa(pid: str) -> int:
    """
    从该受试者 sofa 文件读取 SourcePosition，找到最接近 (0,0) 的方向索引 idx
    """
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        raise FileNotFoundError(f"Missing SOFA for {pid}: {sofa_path}")

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3) [az, el, r]
    az = src_pos[:, 0]
    el = src_pos[:, 1]
    idx = find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, w_el=W_EL)
    return idx


def collect_ids(cache_dir: Path, hrtf_dir: Path):
    """
    只收集同时存在 X、HRTF_512bins、SOFA 的 pid
    """
    x_files = sorted(cache_dir.glob("P*_X.npy"))
    ids = []
    for x in x_files:
        pid = x.name.replace("_X.npy", "")
        h_path = hrtf_dir / f"{pid}{HRTF_SUFFIX}"
        s_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
        if h_path.exists() and s_path.exists():
            ids.append(pid)
    return ids


# =========================
# Dataset
# =========================
class FrontHRTF512BinsDataset(Dataset):
    """
    X: dataset/cache/Pxxxx_X.npy -> (N,3) float32
    H: dataset/HRTF_UpperBins/Pxxxx_HRTF_512bins.npy -> (M,2,512) complex64
    取最接近 (az=0, el=0) 的方向 idx：
      y_raw = H[idx, :, :] -> (2,512) complex
      y = log-magnitude(dB) -> (2,512) float
      -> transpose to (512,2) -> flatten to (1024,)
    """
    def __init__(self, cache_dir: Path, hrtf_dir: Path, ids, eps=1e-12):
        self.cache_dir = cache_dir
        self.hrtf_dir = hrtf_dir
        self.ids = ids
        self.eps = eps

        # 预先为每个 pid 计算 front idx，避免训练时重复读 sofa
        self.front_idx = {}
        for pid in self.ids:
            self.front_idx[pid] = get_front_index_from_sofa(pid)

        # infer out_dim from first sample
        pid0 = self.ids[0]
        H0 = np.load(self.hrtf_dir / f"{pid0}{HRTF_SUFFIX}")
        if H0.ndim != 3 or H0.shape[1] != 2:
            raise ValueError(f"{pid0} HRTF shape invalid: {H0.shape}, expected (M,2,F)")
        self.F = int(H0.shape[2])
        self.out_dim = int(self.F * 2)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        X = np.load(self.cache_dir / f"{pid}_X.npy").astype(np.float32)  # (N,3)

        H = np.load(self.hrtf_dir / f"{pid}{HRTF_SUFFIX}")              # (M,2,F) complex
        if H.ndim != 3 or H.shape[1] != 2:
            raise ValueError(f"{pid} HRTF shape invalid: {H.shape}, expected (M,2,F)")

        i_front = self.front_idx[pid]
        if i_front < 0 or i_front >= H.shape[0]:
            raise IndexError(f"{pid} front idx out of range: {i_front} for M={H.shape[0]}")

        H_front = H[i_front, :, :]  # (2,F) complex

        mag = np.abs(H_front).astype(np.float32)                         # (2,F)
        logmag_db = 20.0 * np.log10(np.maximum(mag, self.eps))           # (2,F)

        # (2,F) -> (F,2) so LSD view(...,-1,2) groups L/R per bin
        y_F2 = np.transpose(logmag_db, (1, 0)).astype(np.float32)        # (F,2)
        y = y_F2.reshape(-1)                                             # (2F,)

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


@torch.no_grad()
def plot_random_3_subjects(model, dataset, n_plot=3):
    """
    随机挑 n_plot 个受试者，画 真值 vs 预测（左/右耳），不打点
    """
    model.eval()

    n_plot = min(n_plot, len(dataset))
    pick = random.sample(range(len(dataset)), k=n_plot)

    for k, idx in enumerate(pick, 1):
        X, y_true, pid = dataset[idx]
        X = X.unsqueeze(0).to(DEVICE)              # (1,N,3)
        y_true = y_true.cpu().numpy()              # (D,)
        y_pred = model(X).squeeze(0).cpu().numpy() # (D,)

        # reshape to (F,2)
        F = dataset.F
        yt = y_true.reshape(F, 2)
        yp = y_pred.reshape(F, 2)

        # 频率轴（仅用于画图，不保存）
        # 你是 512 bins -> NFFT = 1022，sr=48000
        sr = 48000.0
        nfft = 2 * (F - 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

        plt.figure(figsize=(9, 5))
        plt.plot(freqs, yt[:, 0], label="GT Left")
        plt.plot(freqs, yp[:, 0], label="Pred Left")
        plt.plot(freqs, yt[:, 1], label="GT Right")
        plt.plot(freqs, yp[:, 1], label="Pred Right")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Log-magnitude (dB)")
        plt.title(f"[{k}/{n_plot}] {pid} | Front (az=0, el=0) HRTF")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    set_seed(SEED)

    ids = collect_ids(CACHE_DIR, HRTF_DIR)
    if len(ids) < 2:
        raise RuntimeError(
            f"Not enough pairs. Need:\n"
            f"- {CACHE_DIR}/Pxxxx_X.npy\n"
            f"- {HRTF_DIR}/Pxxxx{HRTF_SUFFIX}\n"
            f"- {SOFA_DIR}/Pxxxx{SOFA_SUFFIX}\n"
            f"Found ids={len(ids)}"
        )

    print(f"[Data] Found {len(ids)} subjects with X + HRTF512 + SOFA")
    print("Device:", DEVICE)

    dataset = FrontHRTF512BinsDataset(CACHE_DIR, HRTF_DIR, ids, eps=EPS)
    out_dim = dataset.out_dim
    print("[Dataset] bins(F) =", dataset.F, "| out_dim =", out_dim, "| target=(0,0)")

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

    best_val = float("inf")
    bad_epochs = 0

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "train_lsd", "val_mse", "val_lsd", "bad_epochs"])

    for epoch in range(1, EPOCHS + 1):
        tr_mse, tr_lsd = train_one_epoch(model, train_loader, opt)
        va_mse, va_lsd = eval_one_epoch(model, val_loader)

        improved = (best_val - va_lsd) > MIN_DELTA
        if improved:
            best_val = va_lsd
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "out_dim": out_dim,
                    "best_val_lsd": best_val,
                    "ids": ids,
                    "bins": dataset.F,
                    "target_az": TARGET_AZ,
                    "target_el": TARGET_EL,
                },
                CKPT_OUT
            )
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | train MSE={tr_mse:.4f} LSD={tr_lsd:.3f} | "
            f"val MSE={va_mse:.4f} LSD={va_lsd:.3f} | bad_epochs={bad_epochs}/{PATIENCE}"
        )

        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, tr_mse, tr_lsd, va_mse, va_lsd, bad_epochs])

        if bad_epochs >= PATIENCE:
            print(
                f"\n[Early Stop] No val_lsd improvement > {MIN_DELTA} for {PATIENCE} consecutive epochs. "
                f"Best val_lsd={best_val:.3f}. Stopping at epoch {epoch}."
            )
            break

    print("\nSaved best:", CKPT_OUT, "best_val_lsd=", best_val)
    print("Saved log:", LOG_CSV)

    # ========== 训练结束后：加载 best ckpt 再画 3 个受试者 ==========
    ckpt = torch.load(CKPT_OUT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print("\n[Plot] Random 3 subjects: GT vs Pred (no markers)")
    plot_random_3_subjects(model, dataset, n_plot=N_PLOT_SUBJECTS)


if __name__ == "__main__":
    main()
