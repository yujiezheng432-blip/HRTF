import csv
import random
from pathlib import Path
import shutil

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
HEADNPY_DIR = Path("dataset/HeadNPY")            # X: Pxxxx_HeadNPY.npy
DIFF_DIR    = Path("F:/赵思培科研/HRTF-Project-Phase1/dataset/HRTF_Difference")    # y: Pxxxx_HRTFdiff.npy
SOFA_DIR    = Path("F:/赵思培科研/HRTF-Project-Phase1/dataset/sofa")

HEADNPY_SUFFIX = "_HeadNPY.npy"
DIFF_SUFFIX    = "_HRTFdiff.npy"
SOFA_SUFFIX    = "_FreeFieldCompMinPhase_48kHz.sofa"

TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0

BATCH_SIZE = 2
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.2

NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出目录 -> Basic8
OUT_DIR = Path("Basic8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENCE = 10
MIN_DELTA = 1e-4

N_TEST = 20
N_PLOT_SUBJECTS = 3

SEED_LIST = [73, 2024]
# SEED_LIST = [7, 11, 21, 37, 42, 58, 73, 91, 123, 2024]



# =========================
# Utils
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx


def get_target_index_from_sofa(pid):
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]

    az = src_pos[:, 0]
    el = src_pos[:, 1]

    return find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, W_EL)


def collect_ids():
    ids = []

    for npy_path in sorted(HEADNPY_DIR.glob(f"P*{HEADNPY_SUFFIX}")):
        pid = npy_path.name.replace(HEADNPY_SUFFIX, "")
        diff_path = DIFF_DIR / f"{pid}{DIFF_SUFFIX}"
        sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"

        if diff_path.exists() and sofa_path.exists():
            ids.append(pid)

    return ids


# =========================
# Dataset
# =========================
class TargetDirHRTFDiffDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids
        self.dir_idx = {pid: get_target_index_from_sofa(pid) for pid in ids}

        pid0 = ids[0]
        d = np.load(DIFF_DIR / f"{pid0}{DIFF_SUFFIX}")

        if d.ndim != 3 or d.shape[1] != 2:
            raise ValueError(f"{pid0} diff shape invalid: {d.shape}, expected (M,2,F)")

        self.F = d.shape[2]
        self.out_dim = self.F * 2

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        # ===== 输入改成 HeadNPY =====
        npy_path = HEADNPY_DIR / f"{pid}{HEADNPY_SUFFIX}"
        X = np.load(npy_path).astype(np.float32)   # (N,3)

        D = np.load(DIFF_DIR / f"{pid}{DIFF_SUFFIX}").astype(np.float32)
        i_dir = self.dir_idx[pid]
        D_dir = D[i_dir, :, :]   # (2,F)

        y_F2 = np.transpose(D_dir, (1, 0)).astype(np.float32)   # (F,2)
        y = y_F2.reshape(-1)

        return torch.from_numpy(X), torch.from_numpy(y), pid


# =========================
# Metric
# =========================
def lsd_db(pred, gt):
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if gt.ndim == 1:
        gt = gt.unsqueeze(0)

    B, D = pred.shape

    pred2 = pred.view(B, -1, 2)
    gt2 = gt.view(B, -1, 2)

    diff = pred2 - gt2
    per_ear = torch.sqrt(torch.mean(diff * diff, dim=1))

    return float(torch.mean(per_ear).cpu().item())


# =========================
# PointNet++ helper functions
# =========================
def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def index_points(points, idx):
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device)

    if idx.ndim == 2:
        return points[batch_indices[:, None], idx]
    elif idx.ndim == 3:
        return points[batch_indices[:, None, None], idx]
    else:
        raise ValueError(f"Unsupported idx ndim: {idx.ndim}")


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = torch.argsort(sqrdists, dim=-1)[:, :, :nsample]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel

        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, points
        )

        new_points = new_points.permute(0, 3, 2, 1)  # [B,C,K,S]
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]     # [B,C,S]
        new_points = new_points.permute(0, 2, 1)     # [B,S,C]

        return new_xyz, new_points


# =========================
# Model: PointNet++
# =========================
class PointNet2Regressor(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=3, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512]
        )

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        xyz = x
        points = None

        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        xyz, points = self.sa3(xyz, points)

        global_feat = torch.max(points, dim=1)[0]
        return self.head(global_feat)


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, opt):
    model.train()
    loss_fn = nn.MSELoss()

    mse_sum, lsd_sum, n = 0, 0, 0

    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        b = X.size(0)
        mse_sum += loss.item() * b
        lsd_sum += lsd_db(pred, y) * b
        n += b

    return mse_sum / n, lsd_sum / n


@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    loss_fn = nn.MSELoss()

    mse_sum, lsd_sum, n = 0, 0, 0

    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, y)

        b = X.size(0)
        mse_sum += loss.item() * b
        lsd_sum += lsd_db(pred, y) * b
        n += b

    return mse_sum / n, lsd_sum / n


# =========================
# Plot
# =========================
@torch.no_grad()
def plot_random_subjects(model, dataset, n_plot=3):
    model.eval()
    n_plot = min(n_plot, len(dataset))
    pick = random.sample(range(len(dataset)), k=n_plot)

    for idx in pick:
        X, y_true, pid = dataset[idx]

        X = X.unsqueeze(0).to(DEVICE)
        y_pred = model(X).cpu().numpy().squeeze()
        y_true = y_true.numpy()

        FreqBins = dataset.F

        yt = y_true.reshape(FreqBins, 2)
        yp = y_pred.reshape(FreqBins, 2)

        freqs = np.arange(FreqBins)

        plt.figure(figsize=(9, 5))
        plt.plot(freqs, yt[:, 0], label="GT L")
        plt.plot(freqs, yp[:, 0], label="Pred L")
        plt.plot(freqs, yt[:, 1], label="GT R")
        plt.plot(freqs, yp[:, 1], label="Pred R")
        plt.title(pid)
        plt.xlabel("Frequency bin")
        plt.ylabel("Diff (dB)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =========================
# Train one seed
# =========================
def train_one_seed(seed, ids_sorted):
    set_seed(seed)

    test_ids = ids_sorted[-N_TEST:]
    trainval_ids = ids_sorted[:-N_TEST]

    trainval_dataset = TargetDirHRTFDiffDataset(trainval_ids)
    test_dataset = TargetDirHRTFDiffDataset(test_ids)

    out_dim = trainval_dataset.out_dim

    n_total = len(trainval_dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_set, val_set = random_split(
        trainval_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = PointNet2Regressor(out_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = 1e9
    bad_epochs = 0

    ckpt_path = OUT_DIR / f"seed_{seed}.pt"

    for epoch in range(EPOCHS):
        tr_mse, tr_lsd = train_one_epoch(model, train_loader, opt)
        va_mse, va_lsd = eval_one_epoch(model, val_loader)

        print(f"[seed={seed}] epoch={epoch:03d} | train_lsd={tr_lsd:.4f} | val_lsd={va_lsd:.4f}")

        if best_val - va_lsd > MIN_DELTA:
            best_val = va_lsd
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    te_mse, te_lsd = eval_one_epoch(model, test_loader)

    return best_val, te_lsd, ckpt_path


# =========================
# Main
# =========================
def main():
    ids = sorted(collect_ids())

    if len(ids) < N_TEST + 2:
        raise RuntimeError(f"Not enough valid subjects: total={len(ids)}, need at least {N_TEST+2}")

    print(f"[Data] total={len(ids)} | test(last {N_TEST})={N_TEST}")
    print(f"[Input] HeadNPY")
    print(f"[Model] PointNet++")
    print(f"[Save] {OUT_DIR}")
    print(f"[Device] {DEVICE}")

    best_seed = None
    best_val = 1e9
    best_ckpt = None

    summary_csv = OUT_DIR / "seed_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "best_val_lsd", "test_lsd", "ckpt"])

    for seed in SEED_LIST:
        print("\n" + "=" * 80)
        print(f"Training seed {seed}")
        print("=" * 80)

        val_lsd, test_lsd, ckpt = train_one_seed(seed, ids)

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([seed, val_lsd, test_lsd, str(ckpt)])

        if val_lsd < best_val:
            best_val = val_lsd
            best_seed = seed
            best_ckpt = ckpt

    print("\nBEST SEED:", best_seed)
    shutil.copy(best_ckpt, OUT_DIR / "best_model.pt")
    print("Saved best model to Basic5/best_model.pt")

    # optional: plot
    test_ids = ids[-N_TEST:]
    test_dataset = TargetDirHRTFDiffDataset(test_ids)

    model = PointNet2Regressor(test_dataset.out_dim).to(DEVICE)
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", map_location=DEVICE))

    print("\n[Plot] Random subjects from TEST set")
    plot_random_subjects(model, test_dataset, n_plot=N_PLOT_SUBJECTS)


if __name__ == "__main__":
    main()