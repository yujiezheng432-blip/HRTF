import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================

# STL_DIR = r"dataset/EarSTL"   # 改成你要遍历的文件夹
# STL_DIR = r"dataset/stl"   # 改成你要遍历的文件夹
STL_DIR = r"dataset/HeadSTL"
# STL_DIR = r"Temporary/test"   # 改成你要遍历的文件夹


MAX_POINTS = 30000               # 每个 STL 最多采样点数


# =========================
# 遍历 STL 文件
# =========================
stl_files = [
    f for f in os.listdir(STL_DIR)
    if f.lower().endswith(".stl")
]

if not stl_files:
    raise RuntimeError(f"No STL files found in {STL_DIR}")

print(f"Found {len(stl_files)} STL files")

for fname in stl_files:
    stl_path = os.path.join(STL_DIR, fname)
    print(f"\nVisualizing: {fname}")

    try:
        mesh = trimesh.load(stl_path, force="mesh")
        V = np.asarray(mesh.vertices)

        if V.shape[0] == 0:
            print(f"  ⚠️ Empty mesh, skip")
            continue

        # 随机采样，避免太慢
        n = min(MAX_POINTS, V.shape[0])
        idx = np.random.choice(V.shape[0], size=n, replace=False)
        P = V[idx]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.3)

        ax.set_title(fname)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

        plt.show()   # 关掉窗口后才会显示下一个

    except Exception as e:
        print(f"  ❌ Failed to visualize {fname}: {e}")
