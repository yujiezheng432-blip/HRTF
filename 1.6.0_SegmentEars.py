import trimesh
import numpy as np
import matplotlib.pyplot as plt


# ======================
# 1. 读取 mesh
# ======================
stl_path = r"Temporary/T/P0001_preprocessed.stl"
mesh = trimesh.load(stl_path, force='mesh')

V = mesh.vertices        # (N, 3)
F = mesh.faces           # (M, 3)

# ======================
# 2. 计算 X 轴范围
# ======================
x_min = V[:, 0].min()
x_max = V[:, 0].max()
x_range = x_max - x_min

print("X range:", x_min, "→", x_max)

# 裁剪比例（经验值，可微调）
ratio = 0.12   # 取左右各 12% 的宽度

left_thresh  = x_min + ratio * x_range
right_thresh = x_max - ratio * x_range

print("Left ear threshold X <", left_thresh)
print("Right ear threshold X >", right_thresh)

# ======================
# 3. 根据三角面“中心 X”来筛选面
# ======================
# 每个 face 的三个顶点
face_vertices = V[F]            # (num_faces, 3, 3)
face_center_x = face_vertices[:, :, 0].mean(axis=1)

left_face_mask  = face_center_x < left_thresh
right_face_mask = face_center_x > right_thresh

left_faces  = F[left_face_mask]
right_faces = F[right_face_mask]

print("Left ear faces:", left_faces.shape)
print("Right ear faces:", right_faces.shape)

# ======================
# 4. 构造左右耳 mesh
# ======================
left_ear_mesh = trimesh.Trimesh(
    vertices=V,
    faces=left_faces,
    process=False
)

right_ear_mesh = trimesh.Trimesh(
    vertices=V,
    faces=right_faces,
    process=False
)

# ======================
# 5. 清理 & 导出
# ======================
left_ear_mesh.remove_unreferenced_vertices()
right_ear_mesh.remove_unreferenced_vertices()

left_ear_mesh.export("P0001_left_ear.stl")
right_ear_mesh.export("P0001_right_ear.stl")

print("Exported: P0001_left_ear.stl, P0001_right_ear.stl")

