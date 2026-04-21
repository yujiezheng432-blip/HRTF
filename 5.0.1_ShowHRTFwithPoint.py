import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# ========= 配置 =========
# PID = "P0001"
PID = "P0005"
# PID = "P0010"


SOFA_PATH = Path(f"dataset/sofa/{PID}_FreeFieldCompMinPhase_48kHz.sofa")
# HRTF_NPY  = Path(f"dataset/HRTF/{PID}_HRTF.npy")
HRTF_NPY  = Path(f"dataset/HRTF_UpperBins/{PID}_HRTF_512bins.npy")
# =======================

def wrap_angle_deg(a):
    return (a + 180) % 360 - 180

def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx

targets = [
    ("Front (0,0)",   0.0,   0.0),
    ("Back (180,0)",  180.0, 0.0),
    ("Left (270,0)",  270.0, 0.0),
    ("Right (90,0)",  90.0,  0.0),
    ("Up (0,90)",     0.0,   90.0),
    ("Down (0,-90)",  0.0,  -90.0),
]

# ----------- 读 SOFA 拿方向 + sr -----------
with h5py.File(SOFA_PATH, "r") as f:
    src_pos = f["SourcePosition"][:]
    sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else None

az = src_pos[:, 0]
el = src_pos[:, 1]

# ----------- 读 HRTF npy -----------
H = np.load(HRTF_NPY)  # (M,2,F)
M, R, F = H.shape

print("HRTF shape (M,R,F):", H.shape)
print("Frequency bins (F):", F)

# 频率轴
if sr is not None:
    n_fft = 2 * (F - 1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    xlab = "Frequency (Hz)"
else:
    freqs = np.arange(F)
    xlab = "FFT bin"

# ----------- 每个方向单独画图 -----------
for name, taz, tel in targets:
    idx = find_nearest_direction(az, el, taz, tel)
    az_i, el_i = float(az[idx]), float(el[idx])

    print(f"{name} 频率点数:", F)

    plt.figure(figsize=(8,5))

    # 左耳
    H_L = H[idx, 0, :]
    mag_L_db = 20 * np.log10(np.abs(H_L) + 1e-12)
    plt.plot(freqs, mag_L_db, linewidth=1.2, label="Left")
    plt.scatter(freqs, mag_L_db, s=10)

    # 右耳
    if R > 1:
        H_R = H[idx, 1, :]
        mag_R_db = 20 * np.log10(np.abs(H_R) + 1e-12)
        plt.plot(freqs, mag_R_db, linewidth=1.2, label="Right")
        plt.scatter(freqs, mag_R_db, s=10)

    plt.xlabel(xlab)
    plt.ylabel("Magnitude (dB)")
    plt.title(f"{name} | matched az/el=({az_i:.1f},{el_i:.1f}), idx={idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
