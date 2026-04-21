import trimesh
import numpy as np
import matplotlib.pyplot as plt

# 读取左右耳 STL
# stl_left_path = r"P0001_left_ear.stl"
# stl_right_path = r"P0001_right_ear.stl"


stl_left_path = r"output_Stage2/P0001_ear_lowY.stl"
stl_right_path = r"output_Stage2/P0001_ear_highY.stl"

meshL = trimesh.load(stl_left_path, force='mesh')
meshR = trimesh.load(stl_right_path, force='mesh')

VL = np.asarray(meshL.vertices)
VR = np.asarray(meshR.vertices)

# 随机采样，避免太慢
nL = min(30000, VL.shape[0])
idxL = np.random.choice(VL.shape[0], size=nL, replace=False)
PL = VL[idxL]

nR = min(30000, VR.shape[0])
idxR = np.random.choice(VR.shape[0], size=nR, replace=False)
PR = VR[idxR]

# 画图
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(PL[:, 0], PL[:, 1], PL[:, 2],
           s=0.3, c='blue', label='Left ear')

ax.scatter(PR[:, 0], PR[:, 1], PR[:, 2],
           s=0.3, c='red', label='Right ear')

ax.set_title("Left and Right Ear (sampled point clouds)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 保证比例一致（非常重要）
ax.set_box_aspect([1, 1, 1])

ax.legend()
plt.show()
