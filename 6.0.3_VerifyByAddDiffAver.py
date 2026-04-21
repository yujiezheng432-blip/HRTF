import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
# PIDS = ["P0066", "P0088", "P0099"]
PIDS = ["P0361"]

HRTF_SUBJ_DIR = Path("dataset/HRTF_UpperBins")   # Pxxxx_HRTF_512bins.npy (M,2,F) complex
DIFF_DIR      = Path("dataset/HRTF_Difference")      # Pxxxx_HRTFdiff.npy (M,2,F) float32 (dB diff)
AVG_DIR       = Path("dataset/HRTF_Average")     # al=x_el=y.npy -> (2,F) complex
SOFA_DIR      = Path("dataset/sofa")             # SourcePosition + SamplingRate

HRTF_SUFFIX = "_HRTF_512bins.npy"
DIFF_SUFFIX = "_HRTFdiff.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

# 选“右侧”方向（空间方向，不是右耳）
TARGET_AZ, TARGET_EL = 90.0, 0.0

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


def parse_avg_filename(fp: Path):
    # "al=0.0_el=0.0.npy" -> (0.0, 0.0)
    stem = fp.stem
    parts = stem.split("_")
    az = float(parts[0].split("=")[1])
    el = float(parts[1].split("=")[1])
    return az, el


def load_avg_map(avg_dir: Path):
    avg_files = sorted(avg_dir.glob("al=*_el=*.npy"))
    if not avg_files:
        raise FileNotFoundError(f"No avg files in {avg_dir} (al=*_el=*.npy)")

    avg_map = {}
    for af in avg_files:
        az, el = parse_avg_filename(af)
        az = float(np.round(az, ROUND_DECIMALS))
        el = float(np.round(el, ROUND_DECIMALS))
        A = np.load(af)  # (2,F) complex
        if A.ndim != 2 or A.shape[0] != 2:
            continue
        avg_map[(az, el)] = A.astype(np.complex64)

    if not avg_map:
        raise RuntimeError("Loaded 0 valid avg files.")

    keys = list(avg_map.keys())
    az_all = np.array([k[0] for k in keys], dtype=np.float32)
    el_all = np.array([k[1] for k in keys], dtype=np.float32)
    return avg_map, keys, az_all, el_all


def get_avg_for_direction(avg_map, avg_keys, avg_az_all, avg_el_all, az, el):
    """
    尝试用 rounding 后的 (az,el) 精确匹配 average 文件。
    如果没有，就用 avg_keys 里的最近方向兜底。
    """
    az_k = float(np.round(az, ROUND_DECIMALS))
    el_k = float(np.round(el, ROUND_DECIMALS))
    key = (az_k, el_k)
    if key in avg_map:
        return avg_map[key], key

    j = find_nearest_direction(avg_az_all, avg_el_all, az_k, el_k, w_el=1.0)
    key2 = avg_keys[j]
    return avg_map[key2], key2


# ========== Load avg lookup ==========
avg_map, avg_keys, avg_az_all, avg_el_all = load_avg_map(AVG_DIR)
print(f"[Avg] Loaded {len(avg_map)} direction averages.")


for pid in PIDS:
    hrtf_path = HRTF_SUBJ_DIR / f"{pid}{HRTF_SUFFIX}"
    diff_path = DIFF_DIR / f"{pid}{DIFF_SUFFIX}"
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"

    if not hrtf_path.exists():
        raise FileNotFoundError(f"Missing subject HRTF: {hrtf_path}")
    if not diff_path.exists():
        raise FileNotFoundError(f"Missing diff file: {diff_path} (generate diff first)")
    if not sofa_path.exists():
        raise FileNotFoundError(f"Missing SOFA: {sofa_path}")

    # --- load subject HRTF complex (M,2,F)
    H = np.load(hrtf_path)
    if H.ndim != 3 or H.shape[1] != 2:
        raise ValueError(f"{pid} bad HRTF shape: {H.shape}, expected (M,2,F)")
    M, _, F = H.shape

    # --- load diff dB (M,2,F)
    diff_db_all = np.load(diff_path)
    if diff_db_all.shape != (M, 2, F):
        raise ValueError(f"{pid} diff shape mismatch: {diff_db_all.shape} vs expected {(M,2,F)}")

    # --- load directions + sr
    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3)
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else SR_FALLBACK

    if src_pos.shape[0] != M:
        raise ValueError(f"{pid} M mismatch: sofa M={src_pos.shape[0]} vs HRTF M={M}")

    az_all = src_pos[:, 0]
    el_all = src_pos[:, 1]

    # --- choose direction idx closest to Right(90,0)
    idx_dir = find_nearest_direction(az_all, el_all, TARGET_AZ, TARGET_EL, w_el=1.0)
    az_i, el_i = float(az_all[idx_dir]), float(el_all[idx_dir])

    # --- average for that matched direction
    A, avg_key = get_avg_for_direction(avg_map, avg_keys, avg_az_all, avg_el_all, az_i, el_i)
    if A.shape != (2, F):
        raise ValueError(f"{pid} avg shape mismatch at key={avg_key}: {A.shape} expected {(2,F)}")

    # --- frequency axis
    nfft = 2 * (F - 1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    # --- compute curves (all in dB, both ears)
    avg_db = 20.0 * np.log10(np.maximum(np.abs(A), EPS))                       # (2,F)
    diff_db = diff_db_all[idx_dir, :, :]                                       # (2,F)
    recon_db = avg_db + diff_db                                                # (2,F)
    orig_db = 20.0 * np.log10(np.maximum(np.abs(H[idx_dir, :, :]), EPS))       # (2,F)

    # --- errors
    err = recon_db - orig_db
    max_err = float(np.max(np.abs(err)))
    mean_err = float(np.mean(np.abs(err)))
    print(f"[Check] {pid} | matched dir=({az_i:.3f},{el_i:.3f}) avg_key={avg_key} | "
          f"max|recon-orig|={max_err:.6f} dB, mean|.|={mean_err:.6f} dB")

    # ========== Plot 2x2 subplots ==========
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)

    # 1) Diff
    axs[0, 0].plot(freqs, diff_db[0], label="Left ear")
    axs[0, 0].plot(freqs, diff_db[1], label="Right ear")
    axs[0, 0].set_title("Diff (dB) = Subj - Avg")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2) Average
    axs[0, 1].plot(freqs, avg_db[0], label="Left ear")
    axs[0, 1].plot(freqs, avg_db[1], label="Right ear")
    axs[0, 1].set_title("Average (dB)")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3) Recon = Diff + Avg
    axs[1, 0].plot(freqs, recon_db[0], label="Left ear")
    axs[1, 0].plot(freqs, recon_db[1], label="Right ear")
    axs[1, 0].set_title("Recon (dB) = Avg + Diff")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4) Original
    axs[1, 1].plot(freqs, orig_db[0], label="Left ear")
    axs[1, 1].plot(freqs, orig_db[1], label="Right ear")
    axs[1, 1].set_title("Original (dB)")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")

    fig.suptitle(
        f"{pid} | Direction=Right(90,0) ~ matched ({az_i:.1f},{el_i:.1f}) | "
        f"AvgKey={avg_key} | maxErr={max_err:.4f} dB",
        fontsize=12
    )
    plt.tight_layout()
    plt.show()