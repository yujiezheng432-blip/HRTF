import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
HRTF_SUBJ_DIR = Path("dataset/HRTF_UpperBins")   # Pxxxx_HRTF_512bins.npy (M,2,F) complex
SOFA_DIR      = Path("dataset/sofa")            # Pxxxx_*.sofa provides SourcePosition
AVG_DIR       = Path("dataset/HRTF_Average")    # al=x_el=y.npy -> (2,F) complex
OUT_DIR       = Path("dataset/HRTF_Difference") # output Pxxxx_HRTFdiff.npy
OUT_DIR.mkdir(parents=True, exist_ok=True)

HRTF_SUFFIX = "_HRTF_512bins.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

ROUND_DECIMALS = 3
EPS = 1e-12
SR_FALLBACK = 48000.0

SHOW_N_SUBJECTS = 3

PLOT_TARGETS = {
    "Front (0,0)": (0.0, 0.0),
    "Right (90,0)": (90.0, 0.0),
    "Left (270,0)": (270.0, 0.0),   # ✅ 用这个做 debug
    "Down (0,-90)": (0.0, -90.0),
}

# ===== DEBUG（以 P0361 为例）=====
DEBUG_PID = "P0361"
DEBUG_TARGET_NAME = "Left (270,0)"   # ✅ al=270.0, el=0.0
PRINT_HEAD = 10   # 打印前N个 + 后N个；None -> 全打印
# =========================


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx


def parse_avg_filename(fp: Path):
    stem = fp.stem
    parts = stem.split("_")
    az = float(parts[0].split("=")[1])
    el = float(parts[1].split("=")[1])
    return az, el


def build_avg_map(avg_dir: Path):
    avg_files = sorted(avg_dir.glob("al=*_el=*.npy"))
    if not avg_files:
        raise FileNotFoundError(f"No average files found in {avg_dir} (al=*_el=*.npy)")

    avg_map = {}
    F_ref = None
    for af in avg_files:
        az, el = parse_avg_filename(af)
        az = float(np.round(az, ROUND_DECIMALS))
        el = float(np.round(el, ROUND_DECIMALS))
        A = np.load(af)
        if A.ndim != 2 or A.shape[0] != 2:
            print(f"[Skip avg] {af.name} bad shape {A.shape}")
            continue
        if F_ref is None:
            F_ref = int(A.shape[1])
        elif int(A.shape[1]) != F_ref:
            raise ValueError(f"Avg bins mismatch: {af.name} F={A.shape[1]} expected {F_ref}")
        avg_map[(az, el)] = A.astype(np.complex64)

    if not avg_map:
        raise RuntimeError("Loaded 0 valid average directions.")

    keys = list(avg_map.keys())
    az_all = np.array([k[0] for k in keys], dtype=np.float32)
    el_all = np.array([k[1] for k in keys], dtype=np.float32)
    return avg_map, keys, az_all, el_all, F_ref


def pid_from_hrtf_file(fp: Path) -> str:
    return fp.name.split("_")[0]


def short_print_array(arr, head=10, name="arr"):
    arr = np.asarray(arr)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if head is None:
        print(arr)
        return
    if arr.size <= 2 * head:
        print(arr)
        return
    print(arr[:head])
    print("...")
    print(arr[-head:])


# ========== 1) Load avg map ==========
avg_map, avg_keys, avg_az_all, avg_el_all, F_ref = build_avg_map(AVG_DIR)
print(f"[Avg] Loaded {len(avg_map)} avg directions. F={F_ref}")

# ========== 2) Iterate subjects ==========
subj_files = sorted(HRTF_SUBJ_DIR.glob(f"P*{HRTF_SUFFIX}"))
if not subj_files:
    raise FileNotFoundError(f"No subject HRTF files in {HRTF_SUBJ_DIR} with *{HRTF_SUFFIX}")
print(f"[Subj] Found {len(subj_files)} subject files.")

plot_subject_entries = []
ok_cnt = 0
skip_cnt = 0
load_exist_cnt = 0

for hf in subj_files:
    pid = pid_from_hrtf_file(hf)
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        print(f"[Skip] {pid} missing SOFA: {sofa_path}")
        skip_cnt += 1
        continue

    # 读 HRTF（raw complex）
    H = np.load(hf)  # (M,2,F) complex
    if H.ndim != 3 or H.shape[1] != 2:
        print(f"[Skip] {pid} bad HRTF shape: {H.shape}")
        skip_cnt += 1
        continue

    M, R, F = H.shape
    if F != F_ref:
        print(f"[Skip] {pid} bins mismatch: subj F={F}, avg F={F_ref}")
        skip_cnt += 1
        continue

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3)
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else SR_FALLBACK

    if src_pos.shape[0] != M:
        print(f"[Skip] {pid} M mismatch: sofa M={src_pos.shape[0]} vs HRTF M={M}")
        skip_cnt += 1
        continue

    az = src_pos[:, 0]
    el = src_pos[:, 1]

    out_path = OUT_DIR / f"{pid}_HRTFdiff.npy"

    # ✅ 改动：存在就加载；不存在才计算并保存
    if out_path.exists():
        diff_db = np.load(out_path).astype(np.float32)
        if diff_db.shape != (M, 2, F):
            print(f"[Skip] {pid} existing diff shape mismatch: {diff_db.shape} expected {(M,2,F)}")
            skip_cnt += 1
            continue
        load_exist_cnt += 1
        print(f"[Load exist] {pid} <- {out_path.name} | shape={diff_db.shape}")
    else:
        diff_db = np.empty((M, 2, F), dtype=np.float32)

        for m in range(M):
            az_k = float(np.round(az[m], ROUND_DECIMALS))
            el_k = float(np.round(el[m], ROUND_DECIMALS))
            key = (az_k, el_k)

            if key not in avg_map:
                j = find_nearest_direction(avg_az_all, avg_el_all, az_k, el_k, w_el=1.0)
                key = avg_keys[j]

            A = avg_map[key]  # (2,F) complex

            subj_db = 20.0 * np.log10(np.maximum(np.abs(H[m, :, :]), EPS))  # (2,F)
            avg_db  = 20.0 * np.log10(np.maximum(np.abs(A), EPS))           # (2,F)
            diff_db[m, :, :] = (subj_db - avg_db).astype(np.float32)

        np.save(out_path, diff_db)
        ok_cnt += 1
        print(f"[OK] {pid} -> {out_path.name} | shape={diff_db.shape} dtype=float32")

    # ===== DEBUG：针对 P0361 打印 Raw/Avg/Diff（左右耳），方向 270,0 =====
    if pid == DEBUG_PID:
        if DEBUG_TARGET_NAME not in PLOT_TARGETS:
            raise KeyError(f"DEBUG_TARGET_NAME={DEBUG_TARGET_NAME} not in PLOT_TARGETS keys={list(PLOT_TARGETS.keys())}")

        taz, tel = PLOT_TARGETS[DEBUG_TARGET_NAME]
        idx_dir = find_nearest_direction(az, el, taz, tel, w_el=1.0)
        az_i, el_i = float(az[idx_dir]), float(el[idx_dir])

        # 找到对应 avg（允许 nearest）
        az_k = float(np.round(az_i, ROUND_DECIMALS))
        el_k = float(np.round(el_i, ROUND_DECIMALS))
        key = (az_k, el_k)
        if key not in avg_map:
            j = find_nearest_direction(avg_az_all, avg_el_all, az_k, el_k, w_el=1.0)
            key = avg_keys[j]
        A = avg_map[key]  # (2,F) complex

        raw_db = 20.0 * np.log10(np.maximum(np.abs(H[idx_dir, :, :]), EPS))  # (2,F)
        avg_db = 20.0 * np.log10(np.maximum(np.abs(A), EPS))                 # (2,F)
        dif_db = diff_db[idx_dir, :, :]                                      # (2,F)

        print("\n" + "-" * 80)
        print(f"[DEBUG] {pid} | {DEBUG_TARGET_NAME} | matched az/el=({az_i:.3f},{el_i:.3f}) idx={idx_dir}")
        print(f"[DEBUG] avg_key used = {key}")
        print("-" * 80)

        print("[LEFT EAR]")
        short_print_array(raw_db[0], head=PRINT_HEAD, name="Raw dB (Left)")
        short_print_array(avg_db[0], head=PRINT_HEAD, name="Avg dB (Left)")
        short_print_array(dif_db[0], head=PRINT_HEAD, name="Diff dB (Left) = Raw - Avg")

        print("-" * 80)

        print("[RIGHT EAR]")
        short_print_array(raw_db[1], head=PRINT_HEAD, name="Raw dB (Right)")
        short_print_array(avg_db[1], head=PRINT_HEAD, name="Avg dB (Right)")
        short_print_array(dif_db[1], head=PRINT_HEAD, name="Diff dB (Right) = Raw - Avg")

        print("-" * 80 + "\n")

    # plot cache: 前 3 个受试者（无论是加载还是新算）都缓存
    if len(plot_subject_entries) < SHOW_N_SUBJECTS:
        nfft = 2 * (F - 1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

        entry = {"pid": pid, "freqs": freqs, "targets": {}}
        for tname, (taz, tel) in PLOT_TARGETS.items():
            idx_dir = find_nearest_direction(az, el, taz, tel, w_el=1.0)
            az_i, el_i = float(az[idx_dir]), float(el[idx_dir])
            magL = diff_db[idx_dir, 0, :]
            magR = diff_db[idx_dir, 1, :]
            entry["targets"][tname] = (magL, magR, az_i, el_i, idx_dir)
        plot_subject_entries.append(entry)

print(f"\nDone. newly_saved={ok_cnt}, loaded_exist={load_exist_cnt}, skipped={skip_cnt}. Output dir: {OUT_DIR.resolve()}")

# ========== 3) Plot first 3 subjects ==========
for entry in plot_subject_entries:
    pid = entry["pid"]
    freqs = entry["freqs"]
    for tname, (magL, magR, az_i, el_i, idx_dir) in entry["targets"].items():
        plt.figure(figsize=(9, 5))
        plt.plot(freqs, magL, label="Diff dB Left")
        plt.plot(freqs, magR, label="Diff dB Right")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("dB difference (Subj - Avg)")
        plt.title(f"{pid} | {tname} | matched az/el=({az_i:.1f},{el_i:.1f}) idx={idx_dir}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()