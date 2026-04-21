import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PID = "P0001"

HRTF_PATH = Path(f"dataset/HRTF_UpperBins/{PID}_HRTF_512bins.npy")          # (M,2,F) complex
SOFA_PATH = Path(f"dataset/sofa/{PID}_FreeFieldCompMinPhase_48kHz.sofa")   # SourcePosition
AVG_DIR   = Path("dataset/HRTF_Average")                                    # al=x_el=y.npy -> (2,F) complex
OUT_DIR   = Path("dataset/HRTF_detail")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUND_DECIMALS = 3
EPS = 1e-12
SR_FALLBACK = 48000.0
# =========================


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180

def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx

def avg_path_from_dir(az, el):
    az = float(np.round(az, ROUND_DECIMALS))
    el = float(np.round(el, ROUND_DECIMALS))
    return AVG_DIR / f"al={az}_el={el}.npy"


# ========== 1) 读 P0001 的 HRTF + 方向 ==========
if not HRTF_PATH.exists():
    raise FileNotFoundError(f"Missing: {HRTF_PATH}")
if not SOFA_PATH.exists():
    raise FileNotFoundError(f"Missing: {SOFA_PATH}")

H = np.load(HRTF_PATH)  # (M,2,F) complex
if H.ndim != 3 or H.shape[1] != 2:
    raise ValueError(f"Bad HRTF shape: {H.shape}, expected (M,2,F)")

M, R, F = H.shape
print("[P0001] HRTF shape:", H.shape, "dtype:", H.dtype)

with h5py.File(SOFA_PATH, "r") as f:
    src_pos = f["SourcePosition"][:]  # (M,3)
    sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else SR_FALLBACK

if src_pos.shape[0] != M:
    raise ValueError(f"M mismatch: sofa M={src_pos.shape[0]} vs HRTF M={M}")

az_all = src_pos[:, 0]
el_all = src_pos[:, 1]


# ========== 2) 逐方向做 dB 差值 ==========
# diff_db: (M,2,F) float32
diff_db = np.empty((M, 2, F), dtype=np.float32)

missing = 0
for m in range(M):
    az = float(np.round(az_all[m], ROUND_DECIMALS))
    el = float(np.round(el_all[m], ROUND_DECIMALS))
    ap = avg_path_from_dir(az, el)

    if not ap.exists():
        missing += 1
        raise FileNotFoundError(
            f"Average file not found for direction (al={az}, el={el}): {ap}\n"
            f"Tip: check ROUND_DECIMALS or how you named average files."
        )

    A = np.load(ap)  # (2,F) complex
    if A.shape != (2, F):
        raise ValueError(f"Avg shape mismatch at {ap.name}: {A.shape}, expected (2,{F})")

    # per-bin magnitude in dB
    subj_db = 20.0 * np.log10(np.maximum(np.abs(H[m, :, :]), EPS))  # (2,F)
    avg_db  = 20.0 * np.log10(np.maximum(np.abs(A), EPS))           # (2,F)

    diff_db[m, :, :] = (subj_db - avg_db).astype(np.float32)

print(f"[OK] Computed diff_db for all {M} directions. missing={missing}")


# ========== 3) 保存 ==========
out_path = OUT_DIR / f"{PID}_HRTFdiff.npy"
np.save(out_path, diff_db)
print("[Saved]", out_path, "| shape:", diff_db.shape, "dtype:", diff_db.dtype)
print("Meaning: diff_db[m, ear, f] = subj_dB - avg_dB at that direction/bin")


# ========== 4) 示例：画 (az=0,el=0) 的差值曲线（左右耳） ==========
idx00 = find_nearest_direction(az_all, el_all, 0.0, 0.0, w_el=1.0)
az_i, el_i = float(az_all[idx00]), float(el_all[idx00])
print(f"[Plot] nearest to (0,0): idx={idx00}, matched (az,el)=({az_i:.3f},{el_i:.3f})")

nfft = 2 * (F - 1)
freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

plt.figure(figsize=(9, 5))
plt.plot(freqs, diff_db[idx00, 0, :], label="Diff dB Left")
plt.plot(freqs, diff_db[idx00, 1, :], label="Diff dB Right")
plt.xlabel("Frequency (Hz)")
plt.ylabel("dB difference (Subj - Avg)")
plt.title(f"{PID} | dB diff at (0,0) ~ matched ({az_i:.1f},{el_i:.1f})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()