# =========================
# eval_lsd_pointnet.py
# =========================
# 用途：
# 1) 读取你训练好的 PointNet 模型 pointnet_single_dir.pt
# 2) 读取一个耳朵 STL + 对应 SOFA
# 3) 在指定方向 (az, el) 上预测 log-magnitude HRTF
# 4) 计算并打印 LSD (dB)
#
# 运行示例（Windows）：
#   python eval_lsd_pointnet.py ^
#     --ckpt "pointnet_single_dir.pt" ^
#     --ear_stl "F:\赵思培科研\HRTF-Project\P0001_ear_lowY.stl" ^
#     --sofa "F:\赵思培科研\HRTF-Project\Temporary\test\P0001_Raw_44kHz.sofa" ^
#     --az 0 --el 0
#
# 备注：
# - 你的训练目标是 log-magnitude（dB），所以 LSD 就是 log 域差的 RMS（dB）。

import argparse
import numpy as np
import h5py
import trimesh
import torch
import torch.nn as nn


# -------------------------
# 角度与方向索引
# -------------------------
def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    return int(np.argmin(daz**2 + delv**2))


# -------------------------
# mesh -> 点云 -> 标准化
# -------------------------
def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, count=num_points)
    return pts.astype(np.float32)


def normalize_pointcloud(P: np.ndarray) -> np.ndarray:
    P = P.astype(np.float32)
    centroid = P.mean(axis=0, keepdims=True)
    P = P - centroid
    scale = np.max(np.linalg.norm(P, axis=1))
    if scale > 0:
        P = P / scale
    return P


# -------------------------
# SOFA -> 单方向 log-magnitude (dB)
# -------------------------
def sofa_single_direction_logmag(
    sofa_path: str,
    target_az: float,
    target_el: float,
    n_fft: int | None = None,
    fmin_hz: float = 200.0,
    fmax_hz: float = 16000.0,
    eps: float = 1e-12,
):
    with h5py.File(sofa_path, "r") as f:
        src = f["SourcePosition"][:]          # (M,3)
        ir = f["Data.IR"][:]                  # (M,2,N)
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else None

    if sr is None:
        raise RuntimeError("Missing Data.SamplingRate in SOFA.")

    az = src[:, 0]
    el = src[:, 1]
    idx = find_nearest_direction(az, el, target_az, target_el)

    left = ir[idx, 0, :].astype(np.float32)
    right = ir[idx, 1, :].astype(np.float32)
    N = left.shape[0]

    if n_fft is None:
        n_fft = N

    HL = np.fft.rfft(left, n=n_fft)
    HR = np.fft.rfft(right, n=n_fft)
    magL = np.abs(HL)
    magR = np.abs(HR)

    logL = 20.0 * np.log10(magL + eps)
    logR = 20.0 * np.log10(magR + eps)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)

    freqs = freqs[band].astype(np.float32)
    y = np.stack([logL[band], logR[band]], axis=1).astype(np.float32)  # (F,2)
    matched = (float(az[idx]), float(el[idx]))
    return y, freqs, matched


# -------------------------
# LSD（dB）
# -------------------------
def compute_lsd_db(pred_logmag, gt_logmag) -> float:
    """
    pred_logmag, gt_logmag:
      - shape (F,2) 或 (F*2,)
    返回：
      - LSD(dB)，先每只耳朵 RMS，再左右平均
    """
    if isinstance(pred_logmag, torch.Tensor):
        pred = pred_logmag.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_logmag)

    if isinstance(gt_logmag, torch.Tensor):
        gt = gt_logmag.detach().cpu().numpy()
    else:
        gt = np.asarray(gt_logmag)

    if pred.ndim == 1:
        pred = pred.reshape(-1, 2)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 2)

    diff = pred - gt
    lsd_per_ear = np.sqrt(np.mean(diff**2, axis=0))  # (2,)
    return float(np.mean(lsd_per_ear))


# -------------------------
# PointNet baseline（与你训练脚本一致）
# -------------------------
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
        feat = self.mlp(x)                # (B,N,emb)
        g, _ = torch.max(feat, dim=1)     # (B,emb)
        return g


class PointNetRegressor(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.enc = PointNetEncoder(emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        g = self.enc(x)
        return self.head(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pointnet_single_dir.pt")
    ap.add_argument("--ear_stl", required=True, help="ear STL path")
    ap.add_argument("--sofa", required=True, help="SOFA path")
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--el", type=float, default=0.0)
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--fmin", type=float, default=200.0)
    ap.add_argument("--fmax", type=float, default=16000.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) 读 GT（确保 out_dim 一致）
    y_gt, freqs, matched = sofa_single_direction_logmag(
        args.sofa, args.az, args.el, fmin_hz=args.fmin, fmax_hz=args.fmax
    )
    out_dim = y_gt.reshape(-1).shape[0]

    # 2) 读耳朵点云
    mesh = trimesh.load(args.ear_stl, force="mesh")
    P = sample_points_from_mesh(mesh, args.num_points)
    P = normalize_pointcloud(P)
    P_t = torch.from_numpy(P).unsqueeze(0).to(args.device)   # (1,N,3)

    # 3) 载入模型
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = PointNetRegressor(out_dim=out_dim, emb_dim=args.emb_dim).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 4) 预测
    with torch.no_grad():
        y_pred = model(P_t).squeeze(0).cpu().numpy()  # (F*2,)

    # 5) LSD
    lsd_db = compute_lsd_db(y_pred, y_gt.reshape(-1))
    print(f"Matched direction in file: az/el = ({matched[0]:.1f}, {matched[1]:.1f})")
    print(f"Freq bins used: F = {len(freqs)}  -> out_dim = {out_dim}")
    print(f"LSD @ target az/el=({args.az:.1f},{args.el:.1f}) = {lsd_db:.3f} dB")


if __name__ == "__main__":
    main()
