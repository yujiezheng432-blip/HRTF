import trimesh
import numpy as np

# ======================
# 1. 读取 mesh
# ======================
stl_path = r"Temporary/T/P0001_preprocessed.stl"
mesh = trimesh.load(stl_path, force='mesh')

V = mesh.vertices        # (N, 3)
F = mesh.faces           # (M, 3)

# ======================
# 2. 计算 Y 轴范围
# ======================
y_min = V[:, 1].min()
y_max = V[:, 1].max()
y_range = y_max - y_min

print("Y range:", y_min, "→", y_max)

# 裁剪比例（经验值，可微调）
ratio = 0.12   # 先用 10%~15% 都很正常

low_thresh  = y_min + ratio * y_range
high_thresh = y_max - ratio * y_range

print("Low-Y ear threshold Y <", low_thresh)
print("High-Y ear threshold Y >", high_thresh)

# ======================
# 3. 根据三角面“中心 Y”来筛选面
# ======================
face_vertices = V[F]                 # (num_faces, 3, 3)
face_center_y = face_vertices[:, :, 1].mean(axis=1)

low_y_mask  = face_center_y < low_thresh
high_y_mask = face_center_y > high_thresh

low_y_faces  = F[low_y_mask]
high_y_faces = F[high_y_mask]

print("Low-Y ear faces:", low_y_faces.shape)
print("High-Y ear faces:", high_y_faces.shape)

# ======================
# 4. 构造两个耳朵 mesh
# ======================
ear_lowY_mesh = trimesh.Trimesh(
    vertices=V,
    faces=low_y_faces,
    process=False
)

ear_highY_mesh = trimesh.Trimesh(
    vertices=V,
    faces=high_y_faces,
    process=False
)

# 清理无引用顶点
ear_lowY_mesh.remove_unreferenced_vertices()
ear_highY_mesh.remove_unreferenced_vertices()

# ======================
# 5. 导出
# ======================
ear_lowY_mesh.export("P0001_ear_lowY.stl")
ear_highY_mesh.export("P0001_ear_highY.stl")

print("Exported: P0001_ear_lowY.stl, P0001_ear_highY.stl")
