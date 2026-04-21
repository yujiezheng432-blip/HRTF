import numpy as np
import h5py
import matplotlib.pyplot as plt

sofa_path = r"Temporary/T/P0001_Raw_44kHz.sofa"

with h5py.File(sofa_path, "r") as f:
    # 读取方向（793,3）
    src_pos = f["SourcePosition"][:]  # 通常是 [azimuth, elevation, distance] 或类似定义

    # 读取 HRIR（Data.IR）
    if "Data.IR" not in f:
        raise KeyError("This SOFA file does not contain 'Data.IR'. Please check keys.")
    ir = f["Data.IR"][:]  # shape usually: (M, R, N) = (directions, ears, samples)

    # 读取采样率（Data.SamplingRate）
    if "Data.SamplingRate" in f:
        sr = float(np.array(f["Data.SamplingRate"][:]).squeeze())
    else:
        sr = None

print("SourcePosition shape:", src_pos.shape)
print("Data.IR shape:", ir.shape)
print("Sampling rate:", sr)

# ---------------------------
# 1) 可视化：方向点分布（azimuth vs elevation）
# ---------------------------
az = src_pos[:, 0]
el = src_pos[:, 1]

plt.figure()
plt.scatter(az, el, s=8)
plt.xlabel("Azimuth")
plt.ylabel("Elevation")
plt.title("Source positions (Azimuth vs Elevation)")
plt.show()

# ---------------------------
# 2) 选一个方向，看 HRIR（时域）
# ---------------------------
# 选一个“最接近正前方”的方向：az≈0, el≈0
idx = int(np.argmin(az**2 + el**2))
print("Chosen direction index:", idx, "az/el:", az[idx], el[idx])

# 推断维度：默认 (M, R, N)
M, R, N = ir.shape[0], ir.shape[1], ir.shape[2]
left = ir[idx, 0, :]
right = ir[idx, 1, :] if R > 1 else None

# 时间轴（秒）
if sr is not None:
    t = np.arange(N) / sr
else:
    t = np.arange(N)

plt.figure()
plt.plot(t, left, label="Left ear")
if right is not None:
    plt.plot(t, right, label="Right ear")
plt.xlabel("Time (s)" if sr is not None else "Sample index")
plt.ylabel("Amplitude")
plt.title("HRIR (time domain) at one direction")
plt.legend()
plt.show()

# ---------------------------
# 3) HRIR -> HRTF 幅度（频域，dB）
# ---------------------------
if sr is None:
    print("No sampling rate found; skipping frequency axis. (You can still plot FFT bins.)")
else:
    # rFFT（只取正频率）
    H_L = np.fft.rfft(left)
    mag_L = np.abs(H_L)

    plt.figure()
    freqs = np.fft.rfftfreq(N, d=1/sr)
    # 转 dB（避免 log(0)）
    plt.plot(freqs, 20*np.log10(mag_L + 1e-12), label="Left |H(f)| (dB)")

    if right is not None:
        H_R = np.fft.rfft(right)
        mag_R = np.abs(H_R)
        plt.plot(freqs, 20*np.log10(mag_R + 1e-12), label="Right |H(f)| (dB)")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("HRTF magnitude (frequency domain) at one direction")
    plt.legend()
    plt.show()
