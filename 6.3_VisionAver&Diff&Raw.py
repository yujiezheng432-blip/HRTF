import numpy as np
import h5py
from pathlib import Path

# ======================
# CONFIG
# ======================
PID = "P0361"   # 改成你的361号格式
TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0

AVG_DIR = Path("dataset/HRTF_Average")
DIFF_DIR = Path("dataset/HRTF_Difference")
HRTF_DIR = Path("dataset/HRTF_UpperBins")
SOFA_DIR = Path("dataset/sofa")

DIFF_SUFFIX = "_HRTFdiff.npy"
# HRTF_SUFFIX = "_HRTF_512bins.npy"
HRTF_SUFFIX = "_HRTF_512bins.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

EPS = 1e-12


# ======================
# Utils
# ======================
def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx


def get_target_dir_idx(pid):
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3)
    az = src_pos[:, 0]
    el = src_pos[:, 1]
    return find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, w_el=W_EL)


# ======================
# 1️⃣ Average
# ======================
avg_file = AVG_DIR / f"al={float(TARGET_AZ)}_el={float(TARGET_EL)}.npy"
if not avg_file.exists():
    raise FileNotFoundError(f"Average file not found: {avg_file}")

A_complex = np.load(avg_file)  # (2,F) complex
avg_db = 20.0 * np.log10(np.maximum(np.abs(A_complex), EPS))  # (2,F)

# ======================
# 2️⃣ Diff
# ======================
diff_file = DIFF_DIR / f"{PID}{DIFF_SUFFIX}"
if not diff_file.exists():
    raise FileNotFoundError(f"Diff file not found: {diff_file}")

D = np.load(diff_file)  # (M,2,F)
dir_idx = get_target_dir_idx(PID)
diff_db = D[dir_idx, :, :]  # (2,F)

# ======================
# 3️⃣ Raw HRTF
# ======================
raw_file = HRTF_DIR / f"{PID}{HRTF_SUFFIX}"
if not raw_file.exists():
    raise FileNotFoundError(f"Raw HRTF file not found: {raw_file}")

H = np.load(raw_file)  # (M,2,F) complex
H_dir = H[dir_idx, :, :]  # (2,F)
raw_db = 20.0 * np.log10(np.maximum(np.abs(H_dir), EPS))  # (2,F)

# ======================
# PRINT
# ======================

print("========== P0361 ==========")
print(f"Matched direction index: {dir_idx}")
print()

print("1️⃣ Average (dB) [Left Ear]:")
print(avg_db[0])
print("1️⃣ Average (dB) [Right Ear]:")
print(avg_db[1])
print()

print("2️⃣ Diff (dB) [Left Ear]:")
print(diff_db[0])
print("2️⃣ Diff (dB) [Right Ear]:")
print(diff_db[1])
print()

print("3️⃣ Raw HRTF (dB) [Left Ear]:")
print(raw_db[0])
print("3️⃣ Raw HRTF (dB) [Right Ear]:")
print(raw_db[1])
print()