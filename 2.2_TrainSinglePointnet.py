import os
import math
import numpy as np
import h5py
import trimesh
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# 1) 小工具：角度环绕差、找最近方向点
# =========================
def wrap_angle_deg(a):
    return (a + 180) % 360 - 180

def find_nearest_direction(az_all, el_all, target_az, target_el):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + delv**2))
    return idx

# =========================
# 2) 小工具：耳朵 STL -> 采样点云 -> 标准化
# =========================
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

# =========================
# 3) 小工具：SOFA -> 取一个方向的 HRTF log-magnitude
# =========================
def sofa_single_direction_logmag(sofa_path: str,
                                target_az: float,
                                target_el: float,
                                n_fft: int = None,
                                fmin_hz: float = 200.0,
                                fmax_hz: float = 16000.0,
                                eps: float = 1e-12):
    """
    返回:
      y: shape (F, 2)  (频点数, 左右耳)
      freqs: shape (F,)
      matched_azel: (az, el)
    """
    with h5py.File(sofa_path, "r") as f:
        src = f["SourcePosition"][:]          # (M,3)
        ir = f["Data.IR"][:]                  # 通常 (M,2,N)
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else None

    if sr is None:
        raise RuntimeError("No sampling rate found in this SOFA (Data.SamplingRate missing).")

    az = src[:, 0]
    el = src[:, 1]
    idx = find_nearest_direction(az, el, target_az, target_el)

    left = ir[idx, 0, :].astype(np.float32)
    right = ir[idx, 1, :].astype(np.float32)
    N = left.shape[0]

    # FFT长度：默认用信号长度
    if n_fft is None:
        n_fft = N

    # 频域
    HL = np.fft.rfft(left, n=n_fft)
    HR = np.fft.rfft(right, n=n_fft)
    magL = np.abs(HL)
    magR = np.abs(HR)

    # log-magnitude（dB）
    logL = 20.0 * np.log10(magL + eps)
    logR = 20.0 * np.log10(magR + eps)

    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)

    # 只保留一个常用频段（可训练更稳定）
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    freqs = freqs[band]
    y = np.stack([logL[band], logR[band]], axis=1)  # (F,2)

    return y.astype(np.float32), freqs.astype(np.float32), (float(az[idx]), float(el[idx]))

# =========================
# 4) Dataset：一个样本 = 一个耳朵 STL + 对应 SOFA -> 单方向 logmag
# =========================
class EarHRTFDataset(Dataset):
    def __init__(self,
                 pairs,
                 num_points=1024,
                 target_az=0.0,
                 target_el=0.0,
                 fmin_hz=200.0,
                 fmax_hz=16000.0):
        """
        pairs: list of dicts, each dict:
          {
            "ear_stl": "path/to/PXXXX_ear_lowY.stl",
            "sofa":    "path/to/PXXXX_Raw_44kHz.sofa"
          }
        """
        self.pairs = pairs
        self.num_points = num_points
        self.target_az = target_az
        self.target_el = target_el
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz

        # 用第一个样本预先确定输出维度（F*2）
        y0, freqs, matched = sofa_single_direction_logmag(
            self.pairs[0]["sofa"], target_az, target_el,
            fmin_hz=fmin_hz, fmax_hz=fmax_hz
        )
        self.freqs = freqs
        self.out_dim = y0.reshape(-1).shape[0]  # (F*2,)

        print(f"[Dataset] Output dimension = {self.out_dim} (F={len(freqs)} * 2 ears)")
        print(f"[Dataset] Example matched az/el = {matched}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ear_path = self.pairs[idx]["ear_stl"]
        sofa_path = self.pairs[idx]["sofa"]

        # ear mesh -> point cloud
        mesh = trimesh.load(ear_path, force="mesh")
        P = sample_points_from_mesh(mesh, self.num_points)
        P = normalize_pointcloud(P)  # (N,3)

        # SOFA -> target logmag (F,2) -> flatten (F*2,)
        y, _, _ = sofa_single_direction_logmag(
            sofa_path, self.target_az, self.target_el,
            fmin_hz=self.fmin_hz, fmax_hz=self.fmax_hz
        )
        y = y.reshape(-1)  # (F*2,)

        # torch
        P = torch.from_numpy(P)          # (N,3)
        y = torch.from_numpy(y)          # (F*2,)
        return P, y

# =========================
# 5) PointNet baseline
# =========================
class PointNetEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        # per-point MLP: (3)->64->128->emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, N, 3)
        return: (B, emb_dim)
        """
        feat = self.mlp(x)                # (B,N,emb_dim)
        global_feat, _ = torch.max(feat, dim=1)  # max pool over points -> (B,emb_dim)
        return global_feat

class PointNetRegressor(nn.Module):
    def __init__(self, out_dim, emb_dim=256):
        super().__init__()
        self.enc = PointNetEncoder(emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        g = self.enc(x)
        y = self.head(g)
        return y

# =========================
# 6) Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    for P, y in loader:
        P = P.to(device)          # (B,N,3)
        y = y.to(device)          # (B,out_dim)
        pred = model(P)
        loss = torch.mean((pred - y) ** 2)   # log-magnitude MSE（等价于 LSD^2 的核心形式）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * P.size(0)
        n += P.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for P, y in loader:
        P = P.to(device)
        y = y.to(device)
        pred = model(P)
        loss = torch.mean((pred - y) ** 2)
        total += loss.item() * P.size(0)
        n += P.size(0)
    return total / max(n, 1)

def main():
    # ====== 你需要改的“配对表” ======
    # 先用一个人/两个人跑通都可以。后面再扩展遍历所有 PXXXX。
    pairs = [
        {
            "ear_stl": r"F:\赵思培科研\HRTF-Project\P0001_ear_lowY.stl",
            "sofa":    r"test/P0001_Raw_44kHz.sofa",
        },
        # 你可以加第二个样本：
        # {"ear_stl": r"P0002_ear_lowY.stl", "sofa": r".../P0002_Raw_44kHz.sofa"},
    ]

    # 目标方向（你也可以改：左/右/后/上）
    target_az, target_el = 0.0, 0.0

    dataset = EarHRTFDataset(
        pairs=pairs,
        num_points=1024,
        target_az=target_az,
        target_el=target_el,
        fmin_hz=200.0,
        fmax_hz=16000.0
    )

    # 简单切分：80% train, 20% val（样本少时只是为了跑通）
    n = len(dataset)
    n_train = max(1, int(0.8 * n))
    n_val = max(0, n - n_train)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False) if n_val > 0 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointNetRegressor(out_dim=dataset.out_dim, emb_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Device:", device)
    print("Train size:", len(train_set), "Val size:", len(val_set))

    for epoch in range(1, 31):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        if val_loader is not None:
            va = eval_one_epoch(model, val_loader, device)
            print(f"Epoch {epoch:02d} | train MSE={tr:.4f} | val MSE={va:.4f}")
        else:
            print(f"Epoch {epoch:02d} | train MSE={tr:.4f}")

    # 保存模型
    torch.save({"model": model.state_dict(), "freqs": dataset.freqs}, "pointnet_single_dir.pt")
    print("Saved: pointnet_single_dir.pt")

if __name__ == "__main__":
    main()
