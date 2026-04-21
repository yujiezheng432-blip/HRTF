# ==========================================================
# 3.1_DownloadSONICOM_ALL_FINAL.py
# ==========================================================

import time
import random
import re
from pathlib import Path
from urllib.parse import quote
import requests


# =========================
# 🔧 CONFIG（只改这里）
# =========================
OUT_DIR = Path("SONICOM_DL")

START_SUBJECT = 1
END_SUBJECT   = 200        # 下载全部就改这里

SLEEP_SECONDS = 0.8
RETRIES = 3

BASE_URL = "https://transfer.ic.ac.uk:9090"
FUNC_URL = f"{BASE_URL}/WebInterface/function/"

# ✅ 已完全确认的真实路径
SOFA_REL    = "HRTF/HRTF/44kHz/{pid}_Raw_44kHz.sofa"
PLUGGED_REL = "SYNTHETIC_HRTF/{pid}_plugged.stl"
PREPROC_REL = "SYNTHETIC_HRTF/{pid}_preprocessed.stl"

# ⚠️ 每次从浏览器 Network 复制最新 cookie
COOKIE = "currentAuth=TNe7; CrushAuth=1770303557860_bp7yrKL1DSBjUmiRddHUKzmT4aTNe7"


# =========================
# 工具函数
# =========================
def parse_cookie_value(cookie_str: str, key: str):
    m = re.search(rf"(?:^|;\s*){re.escape(key)}=([^;]+)", cookie_str)
    return m.group(1) if m else None


def url_encode_path(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return quote(path, safe="")


def is_real_file_response(resp: requests.Response, expected_fname: str) -> bool:
    cd = resp.headers.get("content-disposition", "")
    ctype = resp.headers.get("content-type", "")
    if resp.status_code != 200:
        return False
    if "attachment" in cd.lower() and expected_fname in cd:
        return True
    if "application" in (ctype or "").lower() and "text" not in (ctype or "").lower():
        return True
    return False


def save_stream(resp: requests.Response, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    tmp.replace(out_path)


def crushftp_download(session, dataset_abs_path: str, out_path: Path) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return "exists"

    c2f = parse_cookie_value(COOKIE, "currentAuth")
    if not c2f:
        return "fail: missing currentAuth"

    form_fields = {
        "command": "download",
        "path": url_encode_path(dataset_abs_path),
        "paths": "",
        "random": str(random.random()),
        "c2f": c2f
    }

    multipart = {k: (None, v) for k, v in form_fields.items()}

    headers = {
        "Origin": BASE_URL,
        "Referer": f"{BASE_URL}/",
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
    }

    for _ in range(RETRIES):
        try:
            resp = session.post(
                FUNC_URL,
                files=multipart,
                headers=headers,
                stream=True,
                timeout=240
            )
            if is_real_file_response(resp, out_path.name):
                save_stream(resp, out_path)
                time.sleep(SLEEP_SECONDS)
                return "ok"
        except Exception:
            pass
        time.sleep(1)

    return "fail"


# =========================
# 主程序
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Cookie": COOKIE})

    failed = []

    for i in range(START_SUBJECT, END_SUBJECT + 1):
        pid = f"P{i:04d}"
        print(f"\n=== {pid} ===")

        sofa_abs = f"/2022_SONICOM-HRTF-DATASET/{pid}/" + SOFA_REL.format(pid=pid)
        sofa_out = OUT_DIR / pid / f"{pid}_Raw_44kHz.sofa"
        r = crushftp_download(session, sofa_abs, sofa_out)
        print(f"{sofa_out.name}: {r}")
        if r not in ("ok", "exists"):
            failed.append((pid, sofa_out.name))

        pre_abs = f"/2022_SONICOM-HRTF-DATASET/{pid}/" + PREPROC_REL.format(pid=pid)
        pre_out = OUT_DIR / pid / f"{pid}_preprocessed.stl"
        r = crushftp_download(session, pre_abs, pre_out)
        print(f"{pre_out.name}: {r}")
        if r not in ("ok", "exists"):
            failed.append((pid, pre_out.name))

        plug_abs = f"/2022_SONICOM-HRTF-DATASET/{pid}/" + PLUGGED_REL.format(pid=pid)
        plug_out = OUT_DIR / pid / f"{pid}_plugged.stl"
        r = crushftp_download(session, plug_abs, plug_out)
        print(f"{plug_out.name}: {r}")
        if r not in ("ok", "exists"):
            failed.append((pid, plug_out.name))

    if failed:
        with open(OUT_DIR / "failed_downloads.txt", "w") as f:
            for pid, fname in failed:
                f.write(f"{pid}\t{fname}\n")
        print("\n⚠️ Some downloads failed. See failed_downloads.txt")
    else:
        print("\n✅ All downloads succeeded.")


if __name__ == "__main__":
    main()
