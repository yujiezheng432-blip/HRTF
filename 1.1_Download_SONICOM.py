import requests
import os
from urllib.parse import quote


def download_simple():
    """简化版下载脚本 - 只下载最关键的HRTF文件"""
    base_url = "https://transfer.ic.ac.uk:9090"
    base_path = "2022_SONICOM-HRTF-DATASET"

    # 前10个参与者
    participants = [f"P{str(i).zfill(4)}" for i in range(1, 11)]

    # 只下载最重要的文件（给初学者的推荐文件）
    for pid in participants:
        print(f"\n处理 {pid}...")

        # 关键文件：初学者推荐的3DTI格式文件
        file_path = f"{base_path}/{pid}/HRTF/HRTF/44.1kHz/{pid}_FreeFieldCompMinPhase_NoITD_44kHz.3dti-hrtf"
        encoded_path = quote(file_path, safe="")
        download_url = f"{base_url}/api/v1/download/{encoded_path}"

        # 保存路径
        save_dir = f"SONICOM_HRTF_SIMPLE/{pid}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{pid}_FreeFieldCompMinPhase_NoITD_44kHz.3dti-hrtf"

        try:
            print(f"下载: {pid}_FreeFieldCompMinPhase_NoITD_44kHz.3dti-hrtf")
            response = requests.get(download_url, stream=True, timeout=60)

            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r进度: {percent:.1f}%", end='')

                print(f"\n✓ 下载完成: {save_path}")

                # 尝试下载README
                try:
                    readme_path = f"{base_path}/{pid}/metadata_and_readme/README.txt"
                    readme_url = f"{base_url}/api/v1/download/{quote(readme_path, safe='')}"
                    readme_response = requests.get(readme_url, timeout=30)
                    if readme_response.status_code == 200:
                        with open(f"{save_dir}/README.txt", 'w', encoding='utf-8') as f:
                            f.write(readme_response.text)
                        print("✓ README下载完成")
                except:
                    print("⚠ README下载失败")

            else:
                print(f"✗ 下载失败: HTTP {response.status_code}")

        except Exception as e:
            print(f"✗ 错误: {e}")


if __name__ == "__main__":
    print("开始下载SONICOM HRTF数据集（简化版）")
    print("只下载每个参与者最重要的HRTF文件（3DTI格式）")
    print("=" * 60)

    download_simple()

    print("\n" + "=" * 60)
    print("下载完成！")
    print("文件保存在: ./Temporary/")
    print("\n使用建议:")
    print("1. 使用3DTI Binaural Test Application打开.3dti-hrtf文件")
    print("2. GitHub: https://github.com/3DTune-In/3dti_AudioToolkit")