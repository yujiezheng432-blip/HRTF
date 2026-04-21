import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

url_template = "https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET/{pid}/3DSCAN/{pid}.stl"

START = 1
END = 380

MAX_WORKERS = 16
TIMEOUT = 60
CHUNK = 1024 * 256          # 256KB
PRINT_EVERY_BYTES = 20 * 1024 * 1024  # 每下载 1MB 打印一次（你可以改成 2MB/5MB）


session = requests.Session()  # 复用连接


def download_one(i: int) -> str:
    pid = f"P{i:04d}"
    url = url_template.format(pid=pid)
    filename = f"{pid}.stl"
    save_path = os.path.join(OUT_DIR, filename)
    tmp_path = save_path + ".part"

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return f"skip {filename}"

    try:
        # 让你“看到开始”
        print(f"[START] {filename}")

        with session.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            downloaded = 0
            next_print = PRINT_EVERY_BYTES

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    # 每 1MB 打印一次进度
                    if downloaded >= next_print:
                        if total > 0:
                            pct = downloaded * 100.0 / total
                            print(f"[DL] {filename}: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB ({pct:.1f}%)")
                        else:
                            print(f"[DL] {filename}: {downloaded/1024/1024:.1f}MB")
                        next_print += PRINT_EVERY_BYTES

        os.replace(tmp_path, save_path)
        print(f"[DONE]  {filename}")
        return f"ok   {filename}"

    except Exception as e:
        # 清理残留 part
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
        print(f"[FAIL] {filename}: {type(e).__name__}: {e}")
        return f"fail {filename}"


def main():
    tasks = list(range(START, END + 1))
    results = {"ok": 0, "skip": 0, "fail": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(download_one, i) for i in tasks]

        # 总体进度条：每完成一个文件 +1
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Finished files"):
            msg = fut.result()
            if msg.startswith("ok"):
                results["ok"] += 1
            elif msg.startswith("skip"):
                results["skip"] += 1
            else:
                results["fail"] += 1

    print("\nDone:", results)


if __name__ == "__main__":
    main()
