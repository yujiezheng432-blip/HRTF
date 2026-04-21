import numpy as np
from pathlib import Path

IN_DIR = Path("dataset/HRTF")
OUT_DIR = Path("dataset/HRTF_UpperBins")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 只处理前5个受试者
MAX_SUBJECTS = 380

# 目标 bins：你说想要 512 bins
TARGET_BINS = 512
NFFT_TARGET = 2 * (TARGET_BINS - 1)  # 1022

files = sorted(IN_DIR.glob("P*_HRTF.npy"))[:MAX_SUBJECTS]
if not files:
    raise FileNotFoundError(f"No files found in {IN_DIR} with pattern P*_HRTF.npy")

print(f"[Info] Will process {len(files)} files (first {MAX_SUBJECTS}).")
print(f"[Target] TARGET_BINS={TARGET_BINS}, NFFT_TARGET={NFFT_TARGET}")

for fp in files:
    H = np.load(fp)  # expected (M, R, F) complex
    if H.ndim != 3:
        print(f"[Skip] {fp.name}: unexpected ndim={H.ndim}, shape={H.shape}")
        continue
    if not np.iscomplexobj(H):
        print(f"[Skip] {fp.name}: not complex dtype={H.dtype} (need complex HRTF to preserve phase)")
        continue

    M, R, F = H.shape
    nfft_orig = 2 * (F - 1)

    # 1) 回到时域 HRIR（长度 nfft_orig）
    hrir = np.fft.irfft(H, n=nfft_orig, axis=-1)  # (M,R,nfft_orig), float

    # 2) 用更长 FFT 重新得到更密的频域 bins
    H_up = np.fft.rfft(hrir, n=NFFT_TARGET, axis=-1).astype(np.complex64)  # (M,R,TARGET_BINS)

    # 3) 保存（文件名带 bins）
    out_fp = OUT_DIR / fp.name.replace("_HRTF.npy", f"_HRTF_{TARGET_BINS}bins.npy")
    np.save(out_fp, H_up)

    print(f"[OK] {fp.name}: (M,R,F)=({M},{R},{F}) nfft_orig={nfft_orig} -> up_bins={H_up.shape[-1]} saved: {out_fp}")

print("\nDone.")
