import os
import requests
from tqdm import tqdm

# 创建 dataset 文件夹
os.makedirs("dataset", exist_ok=True)

# URL 模板
url_template_sofa = "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/{pid}/HRTF/HRTF/48kHz/{pid}_FreeFieldCompMinPhase_48kHz.sofa"
url_template_stl = "https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/P0001/SYNTHETIC_HRTF/{pid}_preprocessed.stl"
# url_template = "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/{pid}/HRTF/HRTF/44kHz/{pid}_Raw_44kHz.sofa"


# 下载每个 P0001 到 P0325 的 .sofa 文件
for i in range(1, 326):
    pid = f"P{i:04d}"
    url = url_template_sofa.format(pid=pid)
    Sofa_filename = f"SOFA//{pid}_FreeFieldCompMinPhase_44kHz.sofa"
    save_path = os.path.join("dataset", Sofa_filename)

    if os.path.exists(save_path):
        print(f"已存在: {Sofa_filename}")
        continue

    try:
        print(f"下载中: {Sofa_filename}")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc=Sofa_filename, unit='KB'):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"下载失败 {Sofa_filename}: {e}")


#    下载STL
for i in range(1, 326):
    pid = f"P{i:04d}"
    url = url_template_stl.format(pid=pid)
    STL_filename = f"STL//{pid}_preprocessed.stl"
    save_path = os.path.join("dataset", STL_filename)

    if os.path.exists(save_path):
        print(f"已存在: {STL_filename}")
        continue

    try:
        print(f"下载中: {STL_filename}")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc=STL_filename, unit='KB'):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"下载失败 {STL_filename}: {e}")