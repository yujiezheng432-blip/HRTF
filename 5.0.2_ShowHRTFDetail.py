import numpy as np
from pathlib import Path

# 修改成你要检查的文件
HRTF_PATH = Path("dataset/HRTF/P0001_HRTF.npy")

H = np.load(HRTF_PATH)

print("=" * 60)
print("File:", HRTF_PATH)
print("dtype:", H.dtype)
print("ndim:", H.ndim)
print("shape:", H.shape)
print("=" * 60)

# ---------- 判断是否复数 ----------
print("Is complex:", np.iscomplexobj(H))

# ---------- 维度含义猜测 ----------
if H.ndim == 3:
    M, R, F = H.shape
    print(f"推测结构: (M, R, F)")
    print("M (方向数量):", M)
    print("R (耳朵数量):", R)
    print("F (frequency bins):", F)
elif H.ndim == 2:
    print("推测结构: (?, F)")
elif H.ndim == 1:
    print("推测结构: (D,) 一维flatten向量")
else:
    print("未知结构")

print("=" * 60)

# ---------- 频率点数 ----------
if H.ndim >= 1:
    print("Frequency bins (最后一维长度):", H.shape[-1])

# ---------- 数值范围 ----------
if np.iscomplexobj(H):
    real_part = H.real
    imag_part = H.imag
    magnitude = np.abs(H)

    print("Real part range:  ", real_part.min(), "→", real_part.max())
    print("Imag part range:  ", imag_part.min(), "→", imag_part.max())
    print("Magnitude range:  ", magnitude.min(), "→", magnitude.max())
else:
    print("Value range:", H.min(), "→", H.max())

print("=" * 60)

# ---------- 打印前几个数 ----------
print("前5个元素示例：")
print(H.flatten()[:5])

print("=" * 60)
