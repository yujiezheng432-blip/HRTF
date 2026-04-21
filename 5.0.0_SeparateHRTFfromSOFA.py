import re
import sys
import numpy as np
from pathlib import Path
from pysofaconventions import SOFAFile

SOFA_DIR = Path("dataset/sofa")
OUT_HRIR_DIR = Path("dataset/HRIR")
OUT_HRTF_DIR = Path("dataset/HRTF")

PATTERN = "P*_FreeFieldCompMinPhase_48kHz.sofa"
N_FFT = None  # 使用原始 HRIR 长度


def extract_pid(path: Path) -> str:
    m = re.match(r"(P\d+)_", path.name)
    return m.group(1) if m else path.name.split("_")[0]


def load_sofa_data(sofa_path: Path):
    sofa = SOFAFile(str(sofa_path), "r")
    try:
        hrir = sofa.getDataIR()        # (M,R,N)
        fs = sofa.getSamplingRate()
    finally:
        sofa.close()

    hrir = np.asarray(hrir, dtype=np.float32)
    fs = float(np.asarray(fs).reshape(-1)[0])
    return hrir, fs


def compute_hrtf_from_hrir(hrir, n_fft=None):
    M, R, N = hrir.shape
    n_fft_use = N if n_fft is None else n_fft

    hrtf = np.fft.rfft(hrir, n=n_fft_use, axis=-1)
    return hrtf.astype(np.complex64)


def main():
    sofa_files = sorted(SOFA_DIR.glob(PATTERN))
    if not sofa_files:
        print("No SOFA files found.")
        sys.exit(1)

    OUT_HRIR_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HRTF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(sofa_files)} SOFA files")

    for sofa_path in sofa_files:
        pid = extract_pid(sofa_path)

        try:
            hrir, fs = load_sofa_data(sofa_path)
            hrtf = compute_hrtf_from_hrir(hrir, N_FFT)

            np.save(OUT_HRIR_DIR / f"{pid}_HRIR.npy", hrir)
            np.save(OUT_HRTF_DIR / f"{pid}_HRTF.npy", hrtf)

            M, R, N = hrir.shape
            F = hrtf.shape[-1]

            print(f"{pid}")
            print(f"Sampling rate: {fs} Hz")
            print(f"HRIR length: {N}")
            print(f"Frequency bins: {F}")
            print("-" * 40)

        except Exception as e:
            print(f"[FAIL] {pid}: {e}")


if __name__ == "__main__":
    main()
