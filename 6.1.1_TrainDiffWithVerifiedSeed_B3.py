import random
from pathlib import Path
import csv
import shutil

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
DIFF_DIR  = Path("dataset/HRTF_Difference")       # y: Pxxxx_HRTFdiff.npy (M,2,F) float32 (dB diff)
SOFA_DIR  = Path("dataset/sofa")                  # 用于 SourcePosition 找目标方向

DIFF_SUFFIX = "_HRTFdiff.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

# 目标方向
TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0

BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2

NUM_WORKERS = 0  # Windows先0稳定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 保存目录
OUT_DIR = Path("Basic3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Early Stopping
PATIENCE = 10
MIN_DELTA = 1e-4

# 测试集：最后 20 个受试者（按 pid 排序）
N_TEST = 20

# 可选：训练结束后画图（从 test 集里抽）
N_PLOT_SUBJECTS = 3

# ✅ 新增：预选 10 个 seeds
# 你也可以改成你想要的一组 seed
SEED_LIST = [7, 11, 21, 37, 42, 58, 73, 91, 123, 2024]

# ✅ best 选择标准：看 val_lsd 最小
# =========================


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让结果更可复现（会稍慢一点）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx


def get_target_index_from_sofa(pid: str) -> int:
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        raise FileNotFoundError(f"Missing SOFA for {pid}: {sofa_path}")

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3) [az, el, r]

    az = src_pos[:, 0]
    el = src_pos[:, 1]
    idx = find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, w_el=W_EL)
    return idx


def collect_ids(cache_dir: Path, diff_dir: Path):
    """
    只收集同时存在 X、DIFF、SOFA 的 pid
    """
    x_files = sorted(cache_dir.glob("P*_X.npy"))
    ids = []
    for x in x_files:
        pid = x.name.replace("_X.npy", "")
        d_path = diff_dir / f"{pid}{DIFF_SUFFIX}"
        s_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
        if d_path.exists() and s_path.exists():
            ids.append(pid)
    return ids


# =========================
# Dataset (Diff label)
# =========================
class TargetDirHRTFDiffDataset(Dataset):
    """
    X: dataset/cache/Pxxxx_X.npy -> (N,3) float32
    y: dataset/HRTF_Difference/Pxxxx_HRTFdiff.npy -> (M,2,F) float32 (dB diff)
       取目标方向 idx：
         y_raw = diff[idx,:,:] -> (2,F)
         -> (F,2) -> flatten -> (2F,)
    """
    def __init__(self, cache_dir: Path, diff_dir: Path, ids):
        self.cache_dir = cache_dir
        self.diff_dir = diff_dir
        self.ids = ids

        # 预先为每个 pid 计算 idx（避免训练时重复读 sofa）
        self.dir_idx = {pid: get_target_index_from_sofa(pid) for pid in self.ids}

        # infer F/out_dim
        pid0 = self.ids[0]
        D0 = np.load(self.diff_dir / f"{pid0}{DIFF_SUFFIX}")
        if D0.ndim != 3 or D0.shape[1] != 2:
            raise ValueError(f"{pid0} diff shape invalid: {D0.shape}, expected (M,2,F)")
        self.F = int(D0.shape[2])
        self.out_dim = int(self.F * 2)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        X = np.load(self.cache_dir / f"{pid}_X.npy").astype(np.float32)  # (N,3)

        D = np.load(self.diff_dir / f"{pid}{DIFF_SUFFIX}").astype(np.float32)  # (M,2,F)
        if D.ndim != 3 or D.shape[1] != 2:
            raise ValueError(f"{pid} diff shape invalid: {D.shape}, expected (M,2,F)")

        i_dir = self.dir_idx[pid]
        if i_dir < 0 or i_dir >= D.shape[0]:
            raise IndexError(f"{pid} dir idx out of range: {i_dir} for M={D.shape[0]}")

        D_dir = D[i_dir, :, :]  # (2,F)

        y_F2 = np.transpose(D_dir, (1, 0)).astype(np.float32)  # (F,2)
        y = y_F2.reshape(-1)                                   # (2F,)
        return torch.from_numpy(X), torch.from_numpy(y), pid


# =========================
# Metric: LSD on (F,2) targets in dB
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
def plot_random_subjects(model, dataset, n_plot=3):
    model.eval()
    n_plot = min(n_plot, len(dataset))
    pick = random.sample(range(len(dataset)), k=n_plot)

    for k, idx in enumerate(pick, 1):
        X, y_true, pid = dataset[idx]
        X = X.unsqueeze(0).to(DEVICE)
        y_true = y_true.cpu().numpy()
        y_pred = model(X).squeeze(0).cpu().numpy()

        F = dataset.F
        yt = y_true.reshape(F, 2)
        yp = y_pred.reshape(F, 2)

        sr = 48000.0
        nfft = 2 * (F - 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

        plt.figure(figsize=(9, 5))
        plt.plot(freqs, yt[:, 0], label="GT Left")
        plt.plot(freqs, yp[:, 0], label="Pred Left")
        plt.plot(freqs, yt[:, 1], label="GT Right")
        plt.plot(freqs, yp[:, 1], label="Pred Right")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Diff (dB)")
        plt.title(f"[{k}/{n_plot}] {pid} | TargetDir (az={TARGET_AZ}, el={TARGET_EL}) Diff")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def train_one_seed(seed: int, ids_sorted):
    """
    对单个 seed 训练一次，返回：
      best_val_lsd, best_val_mse, test_lsd, test_mse, ckpt_path, log_path
    """
    set_seed(seed)

    # 固定 test 为最后 N_TEST 个（不随 seed 变化）
    test_ids = ids_sorted[-N_TEST:]
    trainval_ids = ids_sorted[:-N_TEST]

    # datasets
    trainval_dataset = TargetDirHRTFDiffDataset(CACHE_DIR, DIFF_DIR, trainval_ids)
    test_dataset     = TargetDirHRTFDiffDataset(CACHE_DIR, DIFF_DIR, test_ids)

    out_dim = trainval_dataset.out_dim

    # train/val split（随 seed 变化）
    n_total = len(trainval_dataset)
    n_val = max(1, int(round(n_total * VAL_RATIO)))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        trainval_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # files for this seed
    ckpt_path = OUT_DIR / f"pointnet_diff_seed{seed}.pt"
    log_path  = OUT_DIR / f"train_log_seed{seed}.csv"

    model = PointNetRegressor(out_dim=out_dim, emb_dim=256).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_val_mse = None
    bad_epochs = 0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epoch", "train_mse", "train_lsd", "val_mse", "val_lsd", "bad_epochs"])

    for epoch in range(1, EPOCHS + 1):
        tr_mse, tr_lsd = train_one_epoch(model, train_loader, opt)
        va_mse, va_lsd = eval_one_epoch(model, val_loader)

        improved = (best_val - va_lsd) > MIN_DELTA
        if improved:
            best_val = va_lsd
            best_val_mse = va_mse
            bad_epochs = 0
            torch.save(
                {
                    "seed": seed,
                    "model": model.state_dict(),
                    "out_dim": out_dim,
                    "best_val_lsd": best_val,
                    "best_val_mse": best_val_mse,
                    "trainval_ids": trainval_ids,
                    "test_ids": test_ids,
                    "bins": trainval_dataset.F,
                    "target_az": TARGET_AZ,
                    "target_el": TARGET_EL,
                },
                ckpt_path
            )
        else:
            bad_epochs += 1

        print(
            f"[seed={seed}] Epoch {epoch:03d} | train MSE={tr_mse:.4f} LSD={tr_lsd:.3f} | "
            f"val MSE={va_mse:.4f} LSD={va_lsd:.3f} | bad={bad_epochs}/{PATIENCE}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([seed, epoch, tr_mse, tr_lsd, va_mse, va_lsd, bad_epochs])

        if bad_epochs >= PATIENCE:
            break

    # load best & test eval
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    te_mse, te_lsd = eval_one_epoch(model, test_loader)

    return float(ckpt["best_val_lsd"]), float(ckpt["best_val_mse"]), float(te_lsd), float(te_mse), ckpt_path, log_path


def main():
    ids = collect_ids(CACHE_DIR, DIFF_DIR)
    if len(ids) < (2 + N_TEST):
        raise RuntimeError(
            f"Not enough subjects for train/val/test. Need at least {2 + N_TEST}, got {len(ids)}.\n"
            f"Need:\n"
            f"- {CACHE_DIR}/Pxxxx_X.npy\n"
            f"- {DIFF_DIR}/Pxxxx{DIFF_SUFFIX}\n"
            f"- {SOFA_DIR}/Pxxxx{SOFA_SUFFIX}"
        )

    ids_sorted = sorted(ids)
    print(f"[Data] total={len(ids_sorted)} | test(last {N_TEST})={N_TEST} | device={DEVICE}")
    print("[Seeds]", SEED_LIST)

    # summary file
    summary_csv = OUT_DIR / "seed_sweep_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "best_val_lsd", "best_val_mse", "test_lsd", "test_mse", "ckpt_path", "log_path"])

    best = {
        "seed": None,
        "best_val_lsd": float("inf"),
        "best_val_mse": None,
        "test_lsd": None,
        "test_mse": None,
        "ckpt_path": None,
        "log_path": None,
    }

    for seed in SEED_LIST:
        print("\n" + "=" * 90)
        print(f"Training with seed={seed}")
        print("=" * 90)

        best_val_lsd, best_val_mse, test_lsd, test_mse, ckpt_path, log_path = train_one_seed(seed, ids_sorted)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([seed, best_val_lsd, best_val_mse, test_lsd, test_mse, str(ckpt_path), str(log_path)])

        print(f"[seed={seed}] DONE | best_val_lsd={best_val_lsd:.4f} | test_lsd={test_lsd:.4f} | ckpt={ckpt_path.name}")

        if best_val_lsd < best["best_val_lsd"]:
            best.update(
                seed=seed,
                best_val_lsd=best_val_lsd,
                best_val_mse=best_val_mse,
                test_lsd=test_lsd,
                test_mse=test_mse,
                ckpt_path=ckpt_path,
                log_path=log_path,
            )

    # Save / copy best as canonical file
    best_ckpt_final = OUT_DIR / "pointnet_diff_best.pt"
    shutil.copyfile(best["ckpt_path"], best_ckpt_final)

    # Write a tiny best summary
    best_summary = OUT_DIR / "best_seed_summary.csv"
    with open(best_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["best_seed", "best_val_lsd", "best_val_mse", "test_lsd", "test_mse", "best_ckpt"])
        w.writerow([best["seed"], best["best_val_lsd"], best["best_val_mse"], best["test_lsd"], best["test_mse"], str(best_ckpt_final)])

    print("\n" + "#" * 90)
    print(f"[BEST] seed={best['seed']} | best_val_lsd={best['best_val_lsd']:.4f} | test_lsd={best['test_lsd']:.4f}")
    print(f"[BEST] saved as: {best_ckpt_final}")
    print(f"[BEST] sweep summary: {summary_csv}")
    print(f"[BEST] best summary:  {best_summary}")
    print("#" * 90)

    # （可选）加载 best 模型，在 test 集上画几个例子
    # 这里为了可复现，也 set_seed(best_seed)
    set_seed(int(best["seed"]))
    test_ids = ids_sorted[-N_TEST:]
    test_dataset = TargetDirHRTFDiffDataset(CACHE_DIR, DIFF_DIR, test_ids)

    ckpt = torch.load(best_ckpt_final, map_location=DEVICE)
    model = PointNetRegressor(out_dim=test_dataset.out_dim, emb_dim=256).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print("\n[Plot] Random subjects from TEST set using BEST model")
    plot_random_subjects(model, test_dataset, n_plot=N_PLOT_SUBJECTS)


if __name__ == "__main__":
    main()