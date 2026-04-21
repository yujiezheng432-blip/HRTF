import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
HRTF_DIR = Path("dataset/HRTF_UpperBins")
SOFA_DIR = Path("dataset/sofa")
OUT_DIR  = Path("dataset/HRTF_Average")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HRTF_SUFFIX = "_HRTF_512bins.npy"
SOFA_SUFFIX = "_FreeFieldCompMinPhase_48kHz.sofa"

# 方向 key 的小数保留位数（避免浮点误差导致“同一方向”被拆成多个key）
ROUND_DECIMALS = 3

# 采样率（画频率轴用）
SR = 48000.0

# 要画的 6 个方向
TARGETS = {
    "Front (0,0)":   (0.0,   0.0),
    "Back (180,0)":  (180.0, 0.0),
    "Left (270,0)":  (270.0, 0.0),
    "Right (90,0)":  (90.0,  0.0),
    "Up (0,90)":     (0.0,   90.0),
    "Down (0,-90)":  (0.0,  -90.0),
}
# =========================


def wrap_angle_deg(a):
    return (a + 180) % 360 - 180


def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx


def pid_from_filename(p: Path) -> str:
    # e.g. P0001_HRTF_512bins.npy -> P0001
    name = p.name
    return name.split("_")[0]


# 1) 收集受试者文件
hrtf_files = sorted(HRTF_DIR.glob(f"P*{HRTF_SUFFIX}"))
if not hrtf_files:
    raise FileNotFoundError(f"No HRTF files found: {HRTF_DIR}/P*{HRTF_SUFFIX}")

print(f"[Info] Found {len(hrtf_files)} HRTF files.")


# 2) 遍历累加：每个方向一个 sum + count
sum_map = {}   # key -> complex sum array (2,F)
cnt_map = {}   # key -> count

for hf in hrtf_files:
    pid = pid_from_filename(hf)
    sofa_path = SOFA_DIR / f"{pid}{SOFA_SUFFIX}"
    if not sofa_path.exists():
        print(f"[Skip] Missing sofa for {pid}: {sofa_path}")
        continue

    H = np.load(hf)  # (M,2,F) complex
    if H.ndim != 3 or H.shape[1] != 2:
        print(f"[Skip] Bad HRTF shape for {pid}: {H.shape}")
        continue

    with h5py.File(sofa_path, "r") as f:
        src_pos = f["SourcePosition"][:]  # (M,3) [az, el, r]

    if src_pos.shape[0] != H.shape[0]:
        print(f"[Skip] M mismatch for {pid}: sofa M={src_pos.shape[0]} vs HRTF M={H.shape[0]}")
        continue

    az = src_pos[:, 0]
    el = src_pos[:, 1]

    # 对该受试者每个方向累加
    for m in range(H.shape[0]):
        az_k = float(np.round(az[m], ROUND_DECIMALS))
        el_k = float(np.round(el[m], ROUND_DECIMALS))
        key = (az_k, el_k)

        H_m = H[m, :, :]  # (2,F)
        if key not in sum_map:
            sum_map[key] = np.array(H_m, dtype=np.complex128)  # 累加用更高精度
            cnt_map[key] = 1
        else:
            sum_map[key] += H_m
            cnt_map[key] += 1

print(f"[Info] Collected {len(sum_map)} unique directions (keys).")


# 3) 计算平均并保存
avg_map = {}  # key -> avg (2,F) complex64
for (az_k, el_k), s in sum_map.items():
    c = cnt_map[(az_k, el_k)]
    avg = (s / c).astype(np.complex64)  # (2,F)

    # 按你要求：al=x, el=y 命名（这里 al 写 azimuth）
    out_name = f"al={az_k}_el={el_k}.npy"
    out_path = OUT_DIR / out_name
    np.save(out_path, avg)

    avg_map[(az_k, el_k)] = avg

print(f"[Done] Saved direction-averaged HRTFs to: {OUT_DIR.resolve()}")


# 4) 画 6 个方向（从 avg_map 里找最近方向）
#    为了找最近方向，我们把所有key拆成数组
keys = list(avg_map.keys())
az_all = np.array([k[0] for k in keys], dtype=np.float32)
el_all = np.array([k[1] for k in keys], dtype=np.float32)

# 频率轴：由 bins(F) 推回 NFFT
# 取任意一个 avg 来拿 F
any_avg = next(iter(avg_map.values()))
F = any_avg.shape[-1]
nfft = 2 * (F - 1)
freqs = np.fft.rfftfreq(nfft, d=1.0 / SR)

print(f"[Plot] bins(F)={F}, nfft={nfft}, freq_max={freqs[-1]:.1f} Hz")

for name, (taz, tel) in TARGETS.items():
    idx = find_nearest_direction(az_all, el_all, taz, tel, w_el=1.0)
    az_i = float(az_all[idx])
    el_i = float(el_all[idx])
    key = (az_i, el_i)

    avg = avg_map[key]  # (2,F) complex
    mag_db_L = 20 * np.log10(np.maximum(np.abs(avg[0, :]), 1e-12))
    mag_db_R = 20 * np.log10(np.maximum(np.abs(avg[1, :]), 1e-12))

    plt.figure(figsize=(9, 5))
    plt.plot(freqs, mag_db_L, label="Avg Left")
    plt.plot(freqs, mag_db_R, label="Avg Right")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"{name} | matched al={az_i}, el={el_i} | averaged over {cnt_map[key]} subjects")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()