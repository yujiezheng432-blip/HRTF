import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path

# =========================
# CONFIG
# =========================
CACHE_DIR = Path("dataset/cache")                 # X: Pxxxx_X.npy
DIFF_DIR  = Path("dataset/HRTF_Difference")       # labels exist only for collecting IDs
SOFA_DIR  = Path("dataset/sofa")

DIFF_SUFFIX = "_HRTFdiff.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

# target direction
TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0

# test split
N_TEST = 20

# reference pid
REF_PID = "P0361"

# model checkpoint
CKPT_PATH = Path("Basic3/6.1.1/pointnet_diff_best.pt")   # <= 如果你要用 seed 文件，改成 Basic3/pointnet_diff_seed42.pt 之类

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# choose which ears to include in the absolute-diff-sum
USE_LEFT  = True
USE_RIGHT = True
# =========================


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
        src_pos = f["SourcePosition"][:]  # (M,3)
    az = src_pos[:, 0]
    el = src_pos[:, 1]
    return find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, w_el=W_EL)


def collect_ids(cache_dir: Path, diff_dir: Path):
    """
    只收集同时存在 X、DIFF、SOFA 的 pid（保持与你训练一致的取法）
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
# Model (same as your training)
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


@torch.no_grad()
def predict_diff_db_for_pid(model, pid: str, F: int):
    """
    返回该 pid 在 target direction 上的预测 diff，shape = (2, F) (dB)
    """
    X = np.load(CACHE_DIR / f"{pid}_X.npy").astype(np.float32)  # (N,3)
    X = torch.from_numpy(X).unsqueeze(0).to(DEVICE)             # (1,N,3)

    y_pred_flat = model(X).squeeze(0).detach().cpu().numpy()    # (2F,)
    y_pred_F2 = y_pred_flat.reshape(F, 2)                       # (F,2)
    y_pred_2F = y_pred_F2.T                                     # (2,F)  -> [L; R]
    return y_pred_2F


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing ckpt: {CKPT_PATH}")

    ids = sorted(collect_ids(CACHE_DIR, DIFF_DIR))
    if len(ids) < N_TEST:
        raise RuntimeError(f"Not enough subjects: total={len(ids)} < N_TEST={N_TEST}")

    test_ids = ids[-N_TEST:]
    print(f"[Test Split] total={len(ids)} | test(last {N_TEST})={len(test_ids)}")
    print("Test IDs head/tail:", test_ids[:5], "...", test_ids[-5:])

    if REF_PID not in test_ids:
        raise RuntimeError(f"{REF_PID} is NOT in test set. Your test set is last {N_TEST} of sorted IDs.")

    # infer F (bins) from any diff label file (same as training logic)
    D0 = np.load(DIFF_DIR / f"{test_ids[0]}{DIFF_SUFFIX}")
    if D0.ndim != 3 or D0.shape[1] != 2:
        raise ValueError(f"Bad diff shape for {test_ids[0]}: {D0.shape}, expected (M,2,F)")
    F = int(D0.shape[2])
    out_dim = 2 * F

    # load model
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = PointNetRegressor(out_dim=out_dim, emb_dim=256).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # reference prediction (P0361 at az=270, el=0 direction)
    ref_pred = predict_diff_db_for_pid(model, REF_PID, F)  # (2,F)

    # compute scores
    print("\n[Diff-to-P0361 score]  abs(pred(pid)-pred(P0361)) summed over bins (and ears)")
    print(f"Target direction: az={TARGET_AZ}, el={TARGET_EL} | F={F} bins")
    print(f"Using ears: left={USE_LEFT}, right={USE_RIGHT}\n")

    for pid in test_ids:
        pred = predict_diff_db_for_pid(model, pid, F)  # (2,F)

        # choose ears
        diffs = []
        if USE_LEFT:
            diffs.append(np.abs(pred[0] - ref_pred[0]))   # (F,)
        if USE_RIGHT:
            diffs.append(np.abs(pred[1] - ref_pred[1]))   # (F,)

        score = float(np.sum(diffs))  # scalar
        tag = "(REF)" if pid == REF_PID else ""
        print(f"{pid}: {score:.6f} {tag}")

    print("\nDone.")

if __name__ == "__main__":
    main()