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
DIFF_DIR  = Path("dataset/HRTF_Difference")       # y: Pxxxx_HRTFdiff.npy (M,2,F) float32 (dB diff)
HRTF_DIR  = Path("dataset/HRTF_UpperBins")        # raw HRTF: Pxxxx_HRTF_512bins.npy (M,2,F) complex
AVG_DIR   = Path("dataset/HRTF_Average")          # average: al=xxx_el=yyy.npy (2,F) complex
SOFA_DIR  = Path("dataset/sofa")                  # SourcePosition + SamplingRate

DIFF_SUFFIX = "_HRTFdiff.npy"
HRTF_SUFFIX = "_HRTF_512bins.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

# 目标方向（用于找该受试者的direction idx）
TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0

BATCH_SIZE = 16
VAL_RATIO = 0.2
SEED = 42
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test split: last 20 subjects (sorted)
N_TEST = 20
SHOW_FIRST_K_TEST = 3  # 展示前3个测试受试者（可改）

# 数值稳定
EPS = 1e-12
SR_FALLBACK = 48000.0
ROUND_DECIMALS = 3

# ✅ 只加载 Basic3 下的模型（你按需改名字）
BASIC3_DIR = Path("Basic3/6.1.1")
CKPT_PATH = BASIC3_DIR / "pointnet_diff_best.pt"


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


def get_target_index_from_sofa(pid: str) -> int:
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        raise FileNotFoundError(f"Missing SOFA for {pid}: {sofa_path}")

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3) [az, el, r]
    az = src_pos[:, 0]
    el = src_pos[:, 1]
    return find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, w_el=W_EL)


def get_sofa_positions_and_sr(pid: str):
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        raise FileNotFoundError(f"Missing SOFA for {pid}: {sofa_path}")

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3)
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else SR_FALLBACK
    return src_pos, sr


def collect_ids(cache_dir: Path, diff_dir: Path, hrtf_dir: Path):
    x_files = sorted(cache_dir.glob("P*_X.npy"))
    ids = []
    for x in x_files:
        pid = x.name.replace("_X.npy", "")
        d_path = diff_dir / f"{pid}{DIFF_SUFFIX}"
        h_path = hrtf_dir / f"{pid}{HRTF_SUFFIX}"
        s_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
        if d_path.exists() and h_path.exists() and s_path.exists():
            ids.append(pid)
    return ids


# =========================
# Average map (robust lookup)
# =========================
def parse_avg_filename(fp: Path):
    stem = fp.stem  # "al=270_el=0"
    parts = stem.split("_")
    az = float(parts[0].split("=")[1])
    el = float(parts[1].split("=")[1])
    return az, el


def load_avg_map(avg_dir: Path):
    avg_files = sorted(avg_dir.glob("al=*_el=*.npy"))
    if not avg_files:
        raise FileNotFoundError(f"No average files found in {avg_dir} (al=*_el=*.npy)")

    avg_map = {}
    for af in avg_files:
        az, el = parse_avg_filename(af)
        az = float(np.round(az, ROUND_DECIMALS))
        el = float(np.round(el, ROUND_DECIMALS))
        A = np.load(af)
        if A.ndim != 2 or A.shape[0] != 2:
            continue
        avg_map[(az, el)] = A.astype(np.complex64)

    if not avg_map:
        raise RuntimeError("Loaded 0 valid average files.")

    keys = list(avg_map.keys())
    az_all = np.array([k[0] for k in keys], dtype=np.float32)
    el_all = np.array([k[1] for k in keys], dtype=np.float32)
    return avg_map, keys, az_all, el_all


def get_avg_for_direction(avg_map, avg_keys, avg_az_all, avg_el_all, az, el):
    az_k = float(np.round(az, ROUND_DECIMALS))
    el_k = float(np.round(el, ROUND_DECIMALS))
    key = (az_k, el_k)
    if key in avg_map:
        return avg_map[key], key

    j = find_nearest_direction(avg_az_all, avg_el_all, az_k, el_k, w_el=1.0)
    key2 = avg_keys[j]
    return avg_map[key2], key2


# =========================
# Dataset (Diff label)
# =========================
class TargetDirHRTFDiffDataset(Dataset):
    """
    X: dataset/cache/Pxxxx_X.npy -> (N,3)
    y: dataset/HRTF_Difference/Pxxxx_HRTFdiff.npy -> (M,2,F) dB diff
    取目标方向 idx：y_dir = (2,F) -> (F,2) -> (2F,)
    """
    def __init__(self, cache_dir: Path, diff_dir: Path, ids):
        self.cache_dir = cache_dir
        self.diff_dir = diff_dir
        self.ids = ids

        self.dir_idx = {pid: get_target_index_from_sofa(pid) for pid in self.ids}

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
        i_dir = self.dir_idx[pid]
        D_dir = D[i_dir, :, :]  # (2,F)

        y_F2 = np.transpose(D_dir, (1, 0)).astype(np.float32)  # (F,2)
        y = y_F2.reshape(-1)  # (2F,)
        return torch.from_numpy(X), torch.from_numpy(y), pid


# =========================
# Model
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
# Plot: 3 subplots (Diff L / Diff R / Recon)
# =========================
@torch.no_grad()
def plot_test_predictions_diff_and_recon(model, test_dataset, avg_map_pack, show_k=3):
    """
    对 test_dataset 前 show_k 个受试者：
      - 预测 diff (dB)
      - 读取 270° average（按该受试者实际匹配的 az/el 找最近 average）
      - recon = avg_db + diff
    画 1 个 figure，3 个 subplot：
      图1：左耳 diff 的 GT vs Pred
      图2：右耳 diff 的 GT vs Pred
      图3：加和后的 recon 的 GT vs Pred（双耳都画在一张图里）
    """
    avg_map, avg_keys, avg_az_all, avg_el_all = avg_map_pack
    model.eval()

    show_k = min(show_k, len(test_dataset))
    for i in range(show_k):
        X, y_diff_gt_flat, pid = test_dataset[i]
        Xb = X.unsqueeze(0).to(DEVICE)

        y_diff_pred_flat = model(Xb).squeeze(0).cpu().numpy()  # (2F,)
        y_diff_gt_flat = y_diff_gt_flat.numpy()

        F = test_dataset.F
        diff_pred = y_diff_pred_flat.reshape(F, 2).T  # (2,F)
        diff_gt   = y_diff_gt_flat.reshape(F, 2).T    # (2,F)

        # 方向 & 频率轴
        src_pos, sr = get_sofa_positions_and_sr(pid)
        az_all = src_pos[:, 0]
        el_all = src_pos[:, 1]
        idx_dir = find_nearest_direction(az_all, el_all, TARGET_AZ, TARGET_EL, w_el=W_EL)
        az_i, el_i = float(az_all[idx_dir]), float(el_all[idx_dir])

        nfft = 2 * (F - 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

        # average (complex -> dB)
        A_complex, avg_key = get_avg_for_direction(avg_map, avg_keys, avg_az_all, avg_el_all, az_i, el_i)
        if A_complex.shape != (2, F):
            raise ValueError(f"{pid} avg shape mismatch at key={avg_key}: {A_complex.shape}, expected (2,{F})")
        avg_db = 20.0 * np.log10(np.maximum(np.abs(A_complex), EPS))  # (2,F)

        # recon (dB)
        recon_pred = avg_db + diff_pred  # (2,F)
        recon_gt   = avg_db + diff_gt    # (2,F)

        # ---- 3 subplots in one figure ----
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

        # 图1：左耳 diff
        axs[0].plot(freqs, diff_gt[0], label="GT L diff")
        axs[0].plot(freqs, diff_pred[0], label="Pred L diff")
        axs[0].set_title("Left Ear Diff (dB): GT vs Pred")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].set_ylabel("Diff (dB)")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        # 图2：右耳 diff
        axs[1].plot(freqs, diff_gt[1], label="GT R diff")
        axs[1].plot(freqs, diff_pred[1], label="Pred R diff")
        axs[1].set_title("Right Ear Diff (dB): GT vs Pred")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Diff (dB)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        # 图3：加和后的 recon（双耳都画）
        axs[2].plot(freqs, recon_gt[0], label="GT L (avg+diff)")
        axs[2].plot(freqs, recon_pred[0], label="Pred L (avg+diff)")
        axs[2].plot(freqs, recon_gt[1], label="GT R (avg+diff)")
        axs[2].plot(freqs, recon_pred[1], label="Pred R (avg+diff)")
        axs[2].set_title("Recon (dB) = Avg + Diff: GT vs Pred (Both Ears)")
        axs[2].set_xlabel("Frequency (Hz)")
        axs[2].set_ylabel("Magnitude (dB)")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()

        fig.suptitle(
            f"{pid} | matched dir=({az_i:.1f},{el_i:.1f}) | avg_key={avg_key} | target=({TARGET_AZ},{TARGET_EL})",
            fontsize=11
        )
        plt.tight_layout()
        plt.show()


def main():
    set_seed(SEED)

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ids = collect_ids(CACHE_DIR, DIFF_DIR, HRTF_DIR)
    if len(ids) < (2 + N_TEST):
        raise RuntimeError(f"Not enough subjects. Need at least {2 + N_TEST}, got {len(ids)}")

    ids = sorted(ids)
    test_ids = ids[-N_TEST:]
    trainval_ids = ids[:-N_TEST]

    print(f"[Split] total={len(ids)} | train+val={len(trainval_ids)} | test(last {N_TEST})={len(test_ids)}")
    print("Device:", DEVICE)

    # Load average map
    avg_map_pack = load_avg_map(AVG_DIR)
    print(f"[Avg] Loaded {len(avg_map_pack[0])} direction averages from {AVG_DIR}")

    # Build datasets (we only need test_dataset for this script)
    test_dataset = TargetDirHRTFDiffDataset(CACHE_DIR, DIFF_DIR, test_ids)
    out_dim = test_dataset.out_dim
    print(f"[TestDataset] bins(F)={test_dataset.F} | out_dim={out_dim}")

    # Build model and load ckpt
    model = PointNetRegressor(out_dim=out_dim, emb_dim=256).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    print(f"[Load] Loaded model from {CKPT_PATH}")

    # Predict on test and plot (first K)
    plot_test_predictions_diff_and_recon(model, test_dataset, avg_map_pack, show_k=SHOW_FIRST_K_TEST)


if __name__ == "__main__":
    main()