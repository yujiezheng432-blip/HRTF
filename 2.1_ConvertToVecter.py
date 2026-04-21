import numpy as np
import trimesh
from pathlib import Path

# =========================
# 配置
# =========================
EAR_STL_PATH = r"output_Stage2/P0001_ear_lowY.stl"  # 改成你的耳朵 stl（lowY/highY 都可以）
NUM_POINTS = 1024                      # 固定点数
SEED = 42
OUT_PATH = r"output_Stage2/P0001_ear_lowY_vec.npy"  # 输出向量文件

rng = np.random.default_rng(SEED)

def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """
    从三角网格上“均匀采样”点（比直接抽 vertices 更靠谱）
    返回: (num_points, 3)
    """
    # trimesh 内置：在表面按面积采样
    points, _ = trimesh.sample.sample_surface(mesh, count=num_points)
    return points.astype(np.float32)

def normalize_pointcloud(P: np.ndarray) -> np.ndarray:
    """
    点云标准化：居中 + 缩放到单位球（最大半径=1）
    P: (N,3)
    """
    P = P.astype(np.float32)
    centroid = P.mean(axis=0, keepdims=True)
    P = P - centroid
    scale = np.max(np.linalg.norm(P, axis=1))
    if scale > 0:
        P = P / scale
    return P

def pointcloud_to_vector(P: np.ndarray) -> np.ndarray:
    """
    (N,3) -> (3N,)
    """
    return P.reshape(-1).astype(np.float32)

def main():
    stl_path = Path(EAR_STL_PATH)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    mesh = trimesh.load(stl_path.as_posix(), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded object is not a Trimesh. Check your STL file.")

    print(mesh)
    print("Vertices:", len(mesh.vertices), "Faces:", len(mesh.faces))
    print("Bounds:\n", mesh.bounds)

    # 1) mesh -> 点云（固定NUM_POINTS）
    P = sample_points_from_mesh(mesh, NUM_POINTS)
    print("Sampled point cloud:", P.shape)

    # 2) 标准化
    Pn = normalize_pointcloud(P)

    # 3) 展平成向量
    vec = pointcloud_to_vector(Pn)
    print("Vector shape:", vec.shape)  # (NUM_POINTS*3,)

    # 保存
    np.save(OUT_PATH, vec)
    print("Saved vector to:", OUT_PATH)

    # （可选）同时保存标准化点云，方便你之后复用/可视化
    np.save(OUT_PATH.replace(".npy", "_pc.npy"), Pn)
    print("Saved normalized point cloud to:", OUT_PATH.replace(".npy", "_pc.npy"))

if __name__ == "__main__":
    main()
