import os
import numpy as np
import h5py
import trimesh
from pathlib import Path

# DATASET_DIR = Path("dataset")
# SOFA_DIR = DATASET_DIR / "sofa"
# CACHE_DIR = DATASET_DIR / "cache"
# CACHE_DIR.mkdir(exist_ok=True)

DATASET_DIR = Path("dataset")
STL_DIR = DATASET_DIR / "stl"
SOFA_DIR = DATASET_DIR / "sofa"
CACHE_DIR = DATASET_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


NUM_POINTS = 1024
TARGET_AZ = 0
TARGET_EL = 0
FMIN = 200
FMAX = 16000


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    return int(np.argmin(daz**2 + delv**2))


def extract_logmag(sofa_path):
    with h5py.File(sofa_path, "r") as f:
        src = f["SourcePosition"][:]
        ir = f["Data.IR"][:]
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze())

    az = src[:, 0]
    el = src[:, 1]
    idx = find_nearest_direction(az, el, TARGET_AZ, TARGET_EL)

    left = ir[idx, 0, :]
    right = ir[idx, 1, :]

    HL = np.fft.rfft(left)
    HR = np.fft.rfft(right)

    magL = np.abs(HL)
    magR = np.abs(HR)

    logL = 20*np.log10(magL + 1e-12)
    logR = 20*np.log10(magR + 1e-12)

    freqs = np.fft.rfftfreq(len(left), 1/sr)
    band = (freqs >= FMIN) & (freqs <= FMAX)

    y = np.stack([logL[band], logR[band]], axis=1)
    return y.reshape(-1).astype(np.float32)

#
# subjects = []
# for stl_path in DATASET_DIR.glob("P*.stl"):
#     pid = stl_path.stem
#     sofa_path = SOFA_DIR / f"{pid}_FreeFieldCompMinPhase_48kHz.sofa"
#     if not sofa_path.exists():
#         continue
#
#     print("Processing", pid)

subjects = []

for stl_path in STL_DIR.glob("P*.stl"):
    pid = stl_path.stem
    sofa_path = SOFA_DIR / f"{pid}_FreeFieldCompMinPhase_48kHz.sofa"

    if not sofa_path.exists():
        continue

    print("Processing", pid)

    mesh = trimesh.load(stl_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)

    pts = pts - pts.mean(axis=0)
    pts = pts / np.max(np.linalg.norm(pts, axis=1))

    y = extract_logmag(sofa_path)

    np.save(CACHE_DIR / f"{pid}_X.npy", pts.astype(np.float32))
    np.save(CACHE_DIR / f"{pid}_y.npy", y.astype(np.float32))

    subjects.append(pid)

print("Finished. Total subjects:", len(subjects))
