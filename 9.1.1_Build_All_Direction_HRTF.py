from pathlib import Path
import numpy as np
import h5py
import re

# =========================
# CONFIG
# =========================
SOFA_DIR = Path("dataset/sofa")
OUT_DIR = Path("dataset/HRTF_AllDirection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"
SAVE_COMPLEX64 = True   # True: 保存 complex64，节省空间
OVERWRITE = True        # False: 已存在则跳过


# =========================
# HELPERS
# =========================
def wrap_angle_deg(angle_deg):
    """Wrap angle to [-180, 180)."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def extract_pid(path: Path):
    """
    从文件名中提取 Pxxxx
    例如: P0001_FreeFieldCompMinPhase_48kHz.sofa -> P0001
    """
    m = re.match(r"(P\d{4})", path.stem)
    if not m:
        raise ValueError(f"Cannot extract subject id from filename: {path.name}")
    return m.group(1)

def read_sofa_hrir_and_positions(sofa_path: Path):
    """
    从 SOFA 读取：
    - HRIR: shape (M, 2, N)
    - SourcePosition: shape (M, 3) -> [azimuth, elevation, distance]
    - SamplingRate
    兼容两种常见命名：
    1) 顶层键: 'Data.IR', 'Data.SamplingRate'
    2) 分组键: 'Data/IR', 'Data/SamplingRate'
    """
    with h5py.File(sofa_path, "r") as f:
        keys = list(f.keys())
        print(f"Top-level keys: {keys}")

        # -------------------------
        # SourcePosition
        # -------------------------
        if "SourcePosition" in f:
            source_pos = f["SourcePosition"][()]
        elif "Data.SourcePosition" in f:
            source_pos = f["Data.SourcePosition"][()]
        elif "Data" in f and "SourcePosition" in f["Data"]:
            source_pos = f["Data"]["SourcePosition"][()]
        else:
            raise KeyError(
                f"{sofa_path.name}: cannot find SourcePosition. "
                f"Available keys: {keys}"
            )

        # -------------------------
        # HRIR / Data.IR
        # -------------------------
        if "Data.IR" in f:
            ir = f["Data.IR"][()]
        elif "Data" in f and "IR" in f["Data"]:
            ir = f["Data"]["IR"][()]
        else:
            raise KeyError(
                f"{sofa_path.name}: cannot find HRIR data ('Data.IR' or 'Data/IR'). "
                f"Available keys: {keys}"
            )

        # -------------------------
        # Sampling Rate
        # -------------------------
        if "Data.SamplingRate" in f:
            sr = f["Data.SamplingRate"][()]
        elif "SamplingRate" in f:
            sr = f["SamplingRate"][()]
        elif "Data" in f and "SamplingRate" in f["Data"]:
            sr = f["Data"]["SamplingRate"][()]
        else:
            raise KeyError(
                f"{sofa_path.name}: cannot find SamplingRate. "
                f"Available keys: {keys}"
            )

    source_pos = np.asarray(source_pos)
    ir = np.asarray(ir)
    sr = np.asarray(sr).squeeze()

    if ir.ndim != 3:
        raise ValueError(
            f"{sofa_path.name}: HRIR shape invalid: {ir.shape}, expected (M,2,N)"
        )

    if ir.shape[1] != 2:
        raise ValueError(
            f"{sofa_path.name}: second dim of HRIR must be 2 ears, got {ir.shape}"
        )

    if source_pos.ndim != 2 or source_pos.shape[0] != ir.shape[0]:
        raise ValueError(
            f"{sofa_path.name}: SourcePosition shape {source_pos.shape} "
            f"not matched with HRIR shape {ir.shape}"
        )

    # 只保留前三列 [az, el, dist]
    if source_pos.shape[1] < 3:
        raise ValueError(
            f"{sofa_path.name}: SourcePosition must have at least 3 columns, got {source_pos.shape}"
        )
    source_pos = source_pos[:, :3]

    sr = float(sr)

    return ir, source_pos, sr

def compute_hrtf_from_hrir(ir, sr):
    """
    HRIR -> HRTF
    ir: (M, 2, N)
    return:
        hrtf: (M, 2, F), complex
        freq_bins: (F,)
    """
    n_samples = ir.shape[-1]
    hrtf = np.fft.rfft(ir, axis=-1)
    freq_bins = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    return hrtf, freq_bins


def get_fixed_sort_index(source_pos):
    """
    固定排序：
    1) azimuth wrap到[-180,180)
    2) 按 (azimuth, elevation, distance) 排序
    """
    az = wrap_angle_deg(source_pos[:, 0].astype(np.float64))
    el = source_pos[:, 1].astype(np.float64)
    dist = source_pos[:, 2].astype(np.float64)

    # np.lexsort 最后一个key优先级最高，因此传入顺序要反过来
    sort_idx = np.lexsort((dist, el, az))
    return sort_idx


def flatten_all_direction_hrtf(hrtf_sorted):
    """
    hrtf_sorted: (M, 2, F)

    输出 1D 向量：
    [左耳所有方向所有频点, 右耳所有方向所有频点]
    具体是：
    left_flat = hrtf_sorted[:, 0, :].reshape(-1)
    right_flat = hrtf_sorted[:, 1, :].reshape(-1)
    """
    left_flat = hrtf_sorted[:, 0, :].reshape(-1)
    right_flat = hrtf_sorted[:, 1, :].reshape(-1)
    all_flat = np.concatenate([left_flat, right_flat], axis=0)
    return all_flat


def build_one_subject(sofa_path: Path):
    pid = extract_pid(sofa_path)

    out_meta = OUT_DIR / f"{pid}_DirBins.npy"
    out_hrtf = OUT_DIR / f"{pid}_AllDirectionHRTF.npy"

    if (not OVERWRITE) and out_meta.exists() and out_hrtf.exists():
        print(f"[Skip] {pid} already exists.")
        return

    print(f"========== {pid} ==========")

    # 1) read SOFA
    ir, source_pos, sr = read_sofa_hrir_and_positions(sofa_path)
    print(f"IR shape           : {ir.shape}")          # (M,2,N)
    print(f"SourcePosition     : {source_pos.shape}")  # (M,3)
    print(f"SamplingRate       : {sr}")

    # 2) HRIR -> HRTF
    hrtf, freq_bins = compute_hrtf_from_hrir(ir, sr)
    print(f"HRTF shape         : {hrtf.shape}")        # (M,2,F)
    print(f"Freq bins shape    : {freq_bins.shape}")   # (F,)

    # 3) fixed order
    sort_idx = get_fixed_sort_index(source_pos)
    source_pos_sorted = source_pos[sort_idx]
    hrtf_sorted = hrtf[sort_idx]

    # 4) flatten: left all dirs then right all dirs
    all_direction_hrtf = flatten_all_direction_hrtf(hrtf_sorted)

    if SAVE_COMPLEX64:
        all_direction_hrtf = all_direction_hrtf.astype(np.complex64)
        hrtf_sorted = hrtf_sorted.astype(np.complex64)
    else:
        all_direction_hrtf = all_direction_hrtf.astype(np.complex128)

    # 5) save metadata
    meta = {
        "subject_id": pid,
        "sampling_rate_hz": np.float32(sr),
        "source_position_sorted_deg_m": source_pos_sorted.astype(np.float32),   # (M,3)
        "sort_index": sort_idx.astype(np.int32),                                 # (M,)
        "frequency_bins_hz": freq_bins.astype(np.float32),                       # (F,)
        "hrtf_shape_sorted": np.array(hrtf_sorted.shape, dtype=np.int32),        # (M,2,F)
        "flatten_rule": (
            "Sort directions by (wrapped_azimuth[-180,180), elevation, distance); "
            "then flatten as [left_all_dirs_all_freqs, right_all_dirs_all_freqs]."
        )
    }

    np.save(out_meta, meta, allow_pickle=True)
    np.save(out_hrtf, all_direction_hrtf)

    print(f"Saved meta          -> {out_meta}")
    print(f"Saved AllDir HRTF   -> {out_hrtf}")
    print(f"Flattened shape     : {all_direction_hrtf.shape}")
    print()


def main():
    sofa_files = sorted(SOFA_DIR.glob(f"*{SOFA_SUFFIX}"))
    if not sofa_files:
        raise FileNotFoundError(f"No sofa files found in {SOFA_DIR} with suffix {SOFA_SUFFIX}")

    print(f"Found {len(sofa_files)} SOFA files.\n")

    for sofa_path in sofa_files:
        try:
            build_one_subject(sofa_path)
        except Exception as e:
            print(f"[Error] {sofa_path.name}: {e}\n")

    print("Done.")


if __name__ == "__main__":
    main()
