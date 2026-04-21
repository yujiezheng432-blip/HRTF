# ==========================================================
# 3.1_DownloadFile_fixed.py
# ==========================================================
# 用途：
#   批量下载 SONICOM 数据（Imperial College transfer）
#
# 下载文件（每个 subject）：
#   - PXXXX_plugged.stl
#   - PXXXX_preprocessed.stl
#   - PXXXX_Raw_44kHz.sofa
#
# 使用方法：
#   1) 修改下面【CONFIG 区域】
#   2) 直接运行：
#        python 3.1_DownloadFile_fixed.py
#
# ⚠️ 注意：
# - Cookie 会过期（几小时～一天）
# - 过期后只需要重新复制浏览器里的 cookie，替换 CONFIG 里的 COOKIE 即可

import time
from pathlib import Path
import requests

# =========================
# 🔧 CONFIG（你只需要改这里）
# =========================

# 保存到哪里
OUT_DIR = Path("SONICOM_DL")

# 下载的 subject 范围
START_SUBJECT = 1      # P0001
END_SUBJECT   = 5      # P0005（先小范围测试，成功后再改大）

# Imperial College transfer 后端
BASE_URL = "https://transfer.ic.ac.uk:9090"
FUNC_URL = f"{BASE_URL}/WebInterface/function/"

# 数据集根目录（固定）
DATASET_ROOT = "/2022_SONICOM-HRTF-DATASET"

# 要下载的文件
FILES = [
    "{pid}_plugged.stl",
    "{pid}_preprocessed.stl",
    "{pid}_Raw_44kHz.sofa",
]

# ⚠️【最重要】从浏览器 Network 里复制的 Cookie
COOKIE = "currentAuth=HhkI; CrushAuth=1770252062497_JCGC6oi4x6NnvlGsx2kzmzt7QjHhkI"

# （可选）如果你看到 URL 里有 c2f=xxxx，就填；否则留 None
C2F = None   # 例如 "HhkI"

# 每个文件下载后的等待时间（秒）
SLEEP_SECONDS = 0.8

# 重试次数
RETRIES = 3


# =========================
# 工具函数
# =========================

def is_real_file_response(resp, expected_fname):
    cd = resp.headers.get("content-disposition", "")
    ctype = resp.headers.get("content-type", "")
    if resp.status_code != 200:
        return False
    if "attachment" in cd.lower() and expected_fname in cd:
        return True
    if "application" in (ctype or "").lower() and "html" not in (ctype or "").lower():
        return True
    return False


def save_stream(resp, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    tmp.replace(out_path)


def make_payloads(full_path: str):
    payloads = []

    d = {"command": "download", "path": full_path}
    if C2F:
        d["c2f"] = C2F
    payloads.append(d)

    d = {"command": "download", "file": full_path}
    if C2F:
        d["c2f"] = C2F
    payloads.append(d)

    d = {"command": "downloadFile", "path": full_path}
    if C2F:
        d["c2f"] = C2F
    payloads.append(d)

    return payloads

def download_one(session, pid, fname):
    out_path = OUT_DIR / pid / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        return "exists"

    full_path = f"{DATASET_ROOT}/{pid}/{fname}"
    payloads = make_payloads(full_path)

    headers = {
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/",
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    }

    last_msg = "unknown"
    for attempt in range(1, RETRIES + 1):
        for form in payloads:
            try:
                r = session.post(
                    FUNC_URL,
                    data=form,
                    headers=headers,
                    stream=True,
                    timeout=180,
                )

                # ✅ 成功：真的返回了文件
                if is_real_file_response(r, fname):
                    save_stream(r, out_path)
                    time.sleep(SLEEP_SECONDS)
                    return "ok"

                # ❌ 没进文件流：打印服务器返回内容（关键）
                ctype = (r.headers.get("content-type") or "").lower()
                if "text" in ctype or "json" in ctype:
                    preview = r.content[:500].decode(errors="ignore")
                    print("\n--- Server response (preview) ---")
                    print(preview)
                    print("--- End preview ---\n")
                    last_msg = "server_text_response"
                else:
                    last_msg = f"HTTP {r.status_code} ctype={ctype}"

            except Exception as e:
                last_msg = f"EXC {type(e).__name__}: {e}"

        time.sleep(min(2 * attempt, 10))

    return f"fail: {last_msg}"

# =========================
# 主程序
# =========================

def main():
    print("Saving to:", OUT_DIR.resolve())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Cookie": COOKIE})

    failed = []

    for i in range(START_SUBJECT, END_SUBJECT + 1):
        pid = f"P{i:04d}"
        print(f"\n=== Downloading {pid} ===")

        for tmpl in FILES:
            fname = tmpl.format(pid=pid)
            status = download_one(session, pid, fname)
            print(f"{pid} | {fname} | {status}")
            if status not in ("ok", "exists"):
                failed.append((pid, fname, status))

    if failed:
        fail_txt = OUT_DIR / "failed_downloads.txt"
        with open(fail_txt, "w", encoding="utf-8") as f:
            for pid, fname, status in failed:
                f.write(f"{pid}\t{fname}\t{status}\n")
        print("\n⚠️ Some files failed. See:", fail_txt)
    else:
        print("\n✅ All downloads completed successfully.")


if __name__ == "__main__":
    main()
