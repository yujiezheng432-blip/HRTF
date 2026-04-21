import numpy as np
import h5py
import matplotlib.pyplot as plt

# sofa_path = r"Temporary/test/P0001_Raw_44kHz.sofa"

# sofa_path = r"dataset/P0001_FreeFieldCompMinPhase_48kHz.sofa"
# sofa_path = r"dataset/sofa/P0066_FreeFieldCompMinPhase_48kHz.sofa"
sofa_path = r"dataset/sofa/P0088_FreeFieldCompMinPhase_48kHz.sofa"


def wrap_angle_deg(a):
    """把角度差wrap到[-180, 180]，处理0°和360°的环绕问题"""
    return (a + 180) % 360 - 180

def find_nearest_direction(az_all, el_all, target_az, target_el, w_el=1.0):
    """
    找到最接近目标(az, el)的方向点索引
    w_el: elevation权重（一般1.0就够）
    """
    daz = wrap_angle_deg(az_all - target_az)
    delv = el_all - target_el
    idx = int(np.argmin(daz**2 + (w_el * delv)**2))
    return idx

with h5py.File(sofa_path, "r") as f:
    src_pos = f["SourcePosition"][:]      # (M, 3)
    ir = f["Data.IR"][:]                  # 通常 (M, 2, N)
    sr = float(np.array(f["Data.SamplingRate"][:]).squeeze()) if "Data.SamplingRate" in f else None

az = src_pos[:, 0]
el = src_pos[:, 1]

print("SourcePosition shape:", src_pos.shape)
print("Data.IR shape:", ir.shape)
print("Sampling rate:", sr)

# 你想看的6个方向（目标值）
targets = {
    "Front (0,0)":   (0.0,   0.0),
    "Back (180,0)":  (180.0, 0.0),
    "Left (270,0)":  (270.0, 0.0),  # 等价 -90
    "Right (90,0)":  (90.0,  0.0),
    "Up (0,90)":     (0.0,   90.0),
    "Down (0,-90)":  (0.0,  -90.0),
}

# 遍历每个目标方向并画图
for name, (taz, tel) in targets.items():
    idx = find_nearest_direction(az, el, taz, tel, w_el=1.0)
    az_i, el_i = float(az[idx]), float(el[idx])

    left = ir[idx, 0, :]
    right = ir[idx, 1, :] if ir.shape[1] > 1 else None
    N = left.shape[0]

    # ---------- HRIR（时域） ----------
    if sr is not None:
        t = np.arange(N) / sr
        xlab = "Time (s)"
    else:
        t = np.arange(N)
        xlab = "Sample index"

    plt.figure()
    plt.plot(t, left, label="Left HRIR")
    if right is not None:
        plt.plot(t, right, label="Right HRIR")
    plt.xlabel(xlab)
    plt.ylabel("Amplitude")
    plt.title(f"HRIR - {name} | matched az/el=({az_i:.1f},{el_i:.1f}), idx={idx}")
    plt.legend()
    plt.show()

    # ---------- HRTF 幅度（频域 dB） ----------
    if sr is None:
        # 没有采样率也能画FFT bin，但你这个文件一般会有sr
        H_L = np.fft.rfft(left)
        mag_L_db = 20*np.log10(np.abs(H_L) + 1e-12)

        plt.figure()
        plt.plot(mag_L_db, label="Left |H(f)| (dB)")
        if right is not None:
            H_R = np.fft.rfft(right)
            mag_R_db = 20*np.log10(np.abs(H_R) + 1e-12)
            plt.plot(mag_R_db, label="Right |H(f)| (dB)")
        plt.xlabel("FFT bin")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"HRTF magnitude - {name} | matched az/el=({az_i:.1f},{el_i:.1f}), idx={idx}")
        plt.legend()
        plt.show()
    else:
        freqs = np.fft.rfftfreq(N, d=1/sr)

        H_L = np.fft.rfft(left)
        mag_L_db = 20*np.log10(np.abs(H_L) + 1e-12)

        plt.figure()
        plt.plot(freqs, mag_L_db, label="Left |H(f)| (dB)")
        if right is not None:
            H_R = np.fft.rfft(right)
            mag_R_db = 20*np.log10(np.abs(H_R) + 1e-12)
            plt.plot(freqs, mag_R_db, label="Right |H(f)| (dB)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"HRTF magnitude - {name} | matched az/el=({az_i:.1f},{el_i:.1f}), idx={idx}")
        plt.legend()
        plt.show()
