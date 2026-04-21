
import trimesh
import numpy as np
import matplotlib.pyplot as plt

# stl_path = r"Temporary/test/P0001_preprocessed.stl"

# stl_path = r"Temporary/test/P0001.stl"

stl_path = r"Temporary/0002/P0002.stl"


mesh = trimesh.load(stl_path, force='mesh')

V = np.asarray(mesh.vertices)

# 随机采样，避免太慢
n = min(30000, V.shape[0])
idx = np.random.choice(V.shape[0], size=n, replace=False)
P = V[idx]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2], s=0.3)

ax.set_title("Head + ears (sampled point cloud)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 让比例一致，避免头被拉扁
ax.set_box_aspect([1,1,1])

plt.show()
