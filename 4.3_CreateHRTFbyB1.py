import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
import matplotlib.pyplot as plt


# =========================
# CONFIG（只改这里）
# =========================
CKPT_PATH = "Basic2/az=0.0,el=0.0/pointnet_from_cache.pt"  # 你的模型
STL_DIR = Path("dataset/stl")                 # 你的 stl 文件夹
NUM_POINTS = 1024                             # 必须和你训练时一致（一般是1024）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# PointNet 模型（必须和训练一致）
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
        feat = self.mlp(x)             # (B,N,emb)
        g, _ = torch.max(feat, dim=1)  # (B,emb)
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
# STL -> 点云
# =========================
def mesh_to_pointcloud(stl_path: Path, num_points=1024) -> np.ndarray:
    mesh = trimesh.load(stl_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)

    pts = pts.astype(np.float32)

    # normalize（和你预处理一致：中心化 + 归一化）
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale

    return pts


def main():
    # 1) 随机选一个 STL
    stl_files = sorted(list(STL_DIR.glob("P*.stl")))
    if len(stl_files) == 0:
        raise RuntimeError(f"No STL files found in {STL_DIR}")

    stl_path = random.choice(stl_files)
    print("Random STL:", stl_path)

    # 2) 读 checkpoint，拿到 out_dim
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    out_dim = ckpt["out_dim"]
    print("Model out_dim:", out_dim)

    # 3) 构建模型并加载参数
    model = PointNetRegressor(out_dim=out_dim, emb_dim=256).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 4) STL -> 点云 -> tensor
    P = mesh_to_pointcloud(stl_path, num_points=NUM_POINTS)
    P_t = torch.from_numpy(P).unsqueeze(0).to(DEVICE)  # (1,N,3)

    # 5) 预测
    with torch.no_grad():
        y_pred = model(P_t).squeeze(0).cpu().numpy()  # (D,)

    print("Pred shape:", y_pred.shape)

    # 6) reshape 为 (F,2) 并画图
    y2 = y_pred.reshape(-1, 2)  # (F,2)
    left = y2[:, 0]
    right = y2[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(left, label="Pred Left (logmag dB)")
    plt.plot(right, label="Pred Right (logmag dB)")
    plt.xlabel("Frequency bin index (in your selected band)")
    plt.ylabel("Log magnitude (dB)")
    plt.title(f"Predicted HRTF (single direction, fixed) | {stl_path.stem}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
