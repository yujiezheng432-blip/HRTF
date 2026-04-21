import numpy as np
import h5py
from pathlib import Path

PID = "P0361"
TARGET_AZ = 270.0
TARGET_EL = 0.0
W_EL = 1.0
EPS = 1e-12

AVG_DIR  = Path("dataset/HRTF_Average")
DIFF_DIR = Path("dataset/HRTF_Difference")
HRTF_DIR = Path("dataset/HRTF")          # 你说的 raw 在这里
SOFA_DIR = Path("dataset/sofa")

DIFF_SUFFIX = "_HRTFdiff.npy"
HRTF_SUFFIX = "_HRTF_512bins.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    return int(np.argmin(daz**2 + (w_el * delv)**2))


def get_dir_idx_and_azel(pid: str):
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        raise FileNotFoundError(f"SOFA missing: {sofa_path}")
    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3) az,el,r
    az = src_pos[:, 0]
    el = src_pos[:, 1]
    idx = find_nearest_direction(az, el, TARGET_AZ, TARGET_EL, w_el=W_EL)
    return idx, float(az[idx]), float(el[idx])


def parse_avg_filename(fp: Path):
    # al=270.0_el=0.0.npy
    stem = fp.stem
    parts = stem.split("_")
    az = float(parts[0].split("=")[1])
    el = float(parts[1].split("=")[1])
    return az, el


def load_avg_db_matched(az_t, el_t):
    avg_files = list(AVG_DIR.glob("al=*_el=*.npy"))
    if not avg_files:
        raise FileNotFoundError(f"No avg files in {AVG_DIR}")

    az_list, el_list = [], []
    for f in avg_files:
        az, el = parse_avg_filename(f)
        az_list.append(az)
        el_list.append(el)

    az_arr = np.array(az_list, dtype=np.float32)
    el_arr = np.array(el_list, dtype=np.float32)

    daz = wrap_angle_deg(az_arr - az_t)
    delv = el_arr - el_t
    j = int(np.argmin(daz**2 + delv**2))

    chosen = avg_files[j]
    A = np.load(chosen)

    if A.ndim != 2 or A.shape[0] != 2:
        raise ValueError(f"Avg shape weird: {A.shape} from {chosen.name}")

    # A expected complex OR real+imag packed
    A_c = ensure_complex(A, name="Average")
    avg_db = 20.0 * np.log10(np.maximum(np.abs(A_c), EPS))
    return avg_db, chosen.name


def ensure_complex(arr, name="H"):
    """
    让 arr 变成 complex:
    - 如果本来就是 complex：直接返回
    - 如果最后一维是 2：当作 [real, imag]
    """
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64)

    # 可能是 (..,2) 表示 real/imag
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        real = arr[..., 0]
        imag = arr[..., 1]
        return (real + 1j * imag).astype(np.complex64)

    # 否则就是纯实数（不太像 HRTF complex，但也给你继续跑）
    print(f"[Warn] {name} is not complex and not (...,2) real/imag. dtype={arr.dtype}, shape={arr.shape}")
    return arr.astype(np.complex64)


def load_diff_db(pid: str, dir_idx: int):
    diff_path = DIFF_DIR / f"{pid}{DIFF_SUFFIX}"
    if not diff_path.exists():
        raise FileNotFoundError(f"Diff missing: {diff_path}")
    D = np.load(diff_path)
    if D.ndim != 3 or D.shape[1] != 2:
        raise ValueError(f"Diff shape weird: {D.shape}, expected (M,2,F)")
    return D[dir_idx].astype(np.float32), D.shape[-1]  # (2,F), F


def load_raw_db(pid: str, dir_idx: int, F_expected: int):
    raw_path = HRTF_DIR / f"{pid}{HRTF_SUFFIX}"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw missing: {raw_path}")
    H = np.load(raw_path)

    # 常见: (M,2,F) complex 或 (M,2,F,2) real/imag
    if H.ndim == 4 and H.shape[-1] == 2:
        H = ensure_complex(H, name="RawHRTF")  # -> (M,2,F) complex
    elif H.ndim == 3:
        H = ensure_complex(H, name="RawHRTF")
    else:
        raise ValueError(f"Raw shape weird: {H.shape} (dtype={H.dtype}). Expect (M,2,F) or (M,2,F,2)")

    if H.shape[1] != 2:
        raise ValueError(f"Raw ear dim !=2 : {H.shape}")

    if H.shape[2] != F_expected:
        raise ValueError(f"Raw F mismatch: raw F={H.shape[2]} vs diff F={F_expected}")

    H_dir = H[dir_idx]  # (2,F) complex
    raw_db = 20.0 * np.log10(np.maximum(np.abs(H_dir), EPS)).astype(np.float32)
    return raw_db


def main():
    dir_idx, az_i, el_i = get_dir_idx_and_azel(PID)
    diff_db, F = load_diff_db(PID, dir_idx)
    raw_db = load_raw_db(PID, dir_idx, F_expected=F)
    avg_db, avg_fname = load_avg_db_matched(az_i, el_i)

    print(f"========== {PID} ==========")
    print(f"Matched dir_idx={dir_idx} | actual dir=({az_i:.3f},{el_i:.3f}) | avg_file={avg_fname}")
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

    # ✅ 核对：如果 diff 真的是 dB 差，那么 raw_db - avg_db 应该 ≈ diff_db
    check = raw_db - avg_db
    mae_L = float(np.mean(np.abs(check[0] - diff_db[0])))
    mae_R = float(np.mean(np.abs(check[1] - diff_db[1])))
    mae_all = float(np.mean(np.abs(check - diff_db)))
    print(f"[Check] MAE(raw_db - avg_db  vs diff_db): L={mae_L:.6f}, R={mae_R:.6f}, ALL={mae_all:.6f}")

    # 以及你关心的“avg+diff 是否对上 raw”
    recon = avg_db + diff_db
    mae_recon = float(np.mean(np.abs(recon - raw_db)))
    print(f"[Check] MAE(avg_db + diff_db  vs raw_db): {mae_recon:.6f}")


if __name__ == "__main__":
    main()