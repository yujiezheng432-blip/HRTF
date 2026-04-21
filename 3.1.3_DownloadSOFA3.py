import os
import requests
from tqdm import tqdm

# 创建 dataset 文件夹
os.makedirs("dataset", exist_ok=True)

# URL 模板
url_template = "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/{pid}/HRTF/HRTF/48kHz/{pid}_FreeFieldCompMinPhase_48kHz.sofa"

# 下载每个 P0001 到 P0325 的 .sofa 文件
for i in range(325, 381):
    pid = f"P{i:04d}"
    url = url_template.format(pid=pid)
    filename = f"{pid}_FreeFieldCompMinPhase_48kHz.sofa"
    save_path = os.path.join("dataset", filename)

    if os.path.exists(save_path):
        print(f"已存在: {filename}")
        continue

    try:
        print(f"下载中: {filename}")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc=filename, unit='KB'):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"下载失败 {filename}: {e}")