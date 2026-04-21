import h5py
import numpy as np

sofa_path = r"Temporary/T/P0001_Raw_44kHz.sofa"

with h5py.File(sofa_path, "r") as f:
    # 1️⃣ 看文件里到底有什么
    print("Top-level keys:")
    for k in f.keys():
        print(" -", k)

        # 最常见的：HRIR 在 Data/IR
    if "Data" in f and "IR" in f["Data"]:
        ir = f["Data"]["IR"][:]  # 读成 numpy 数组
        print("Data/IR shape:", ir.shape, "dtype:", ir.dtype)

        # 你可以顺手看看前几个值，确认“它确实是一堆数字”
        print("Sample values:", ir.flatten()[:10])




    print("\n--- Core audio data ---")


    # 方向信息通常在 SourcePosition
    if "SourcePosition" in f:
        sp = f["SourcePosition"][:]
        print("SourcePosition shape:", sp.shape, "dtype:", sp.dtype)


    # 2️⃣ HRIR 数据（最重要）
    if "Data.IR" in f:
        ir = f["Data.IR"][:]
        print("Data.IR shape:", ir.shape)
        print("Data.IR dtype:", ir.dtype)
        print("Sample values:", ir.flatten()[:10])

    # 3️⃣ 采样率（你这个文件用的是这个）
    if "Data.SamplingRate" in f:
        sr = f["Data.SamplingRate"][:]
        print("Data.SamplingRate:", sr)

    # 4️⃣ Delay（现在不用，但可以确认一下）
    if "Data.Delay" in f:
        delay = f["Data.Delay"][:]
        print("Data.Delay shape:", delay.shape)
