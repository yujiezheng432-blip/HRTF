# ==========================================
# delete_bad_sofa.py
# ==========================================
# 功能：
#   自动检测 dataset/sofa 下损坏的 .sofa 文件
#   如果文件打不开 / 缺少关键字段 / 文件过小
#   自动删除
#
# 使用：
#   python delete_bad_sofa.py
# ==========================================

import os
from pathlib import Path
import h5py

SOFA_DIR = Path("dataset/sofa")

# 小于 1MB 基本可以判定为下载失败
MIN_SIZE_BYTES = 1 * 1024 * 1024


def is_valid_sofa(path: Path):
    """
    检查：
    1. 文件能否打开
    2. 是否包含关键字段
    """
    try:
        with h5py.File(path, "r") as f:
            # 必须有这些字段
            if "Data.IR" not in f:
                return False
            if "SourcePosition" not in f:
                return False
        return True
    except:
        return False


def main():
    if not SOFA_DIR.exists():
        print("SOFA folder not found:", SOFA_DIR)
        return

    sofa_files = list(SOFA_DIR.glob("P*.sofa"))
    print(f"Found {len(sofa_files)} sofa files\n")

    deleted = 0
    kept = 0

    for sofa_path in sofa_files:
        size = sofa_path.stat().st_size

        # 1️⃣ 文件太小 → 删除
        if size < MIN_SIZE_BYTES:
            print(f"[DELETE small] {sofa_path.name}  size={size}")
            sofa_path.unlink()
            deleted += 1
            continue

        # 2️⃣ 打不开 / 结构异常 → 删除
        if not is_valid_sofa(sofa_path):
            print(f"[DELETE corrupt] {sofa_path.name}")
            sofa_path.unlink()
            deleted += 1
            continue

        kept += 1

    print("\n==========================")
    print(f"Kept:    {kept}")
    print(f"Deleted: {deleted}")
    print("==========================")


if __name__ == "__main__":
    main()
