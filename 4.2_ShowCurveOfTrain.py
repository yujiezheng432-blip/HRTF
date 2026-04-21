import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 配置
# =========================
CSV_PATH = "Basic2/az=0.0,el=0.0/train_log.csv"
SAVE_FIG = True


# =========================
# 读取 CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("Loaded CSV with columns:", df.columns.tolist())
print("Total epochs:", len(df))


# =========================
# 画 MSE 曲线
# =========================
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_mse"], label="Train MSE")
plt.plot(df["epoch"], df["val_mse"], label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training MSE Curve")
plt.legend()
plt.grid(True)

if SAVE_FIG:
    plt.savefig("mse_curve.png", dpi=300)

plt.show()


# =========================
# 画 LSD 曲线
# =========================
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["train_lsd"], label="Train LSD (dB)")
plt.plot(df["epoch"], df["val_lsd"], label="Val LSD (dB)")
plt.xlabel("Epoch")
plt.ylabel("LSD (dB)")
plt.title("Training LSD Curve")
plt.legend()
plt.grid(True)

if SAVE_FIG:
    plt.savefig("lsd_curve.png", dpi=300)

plt.show()
