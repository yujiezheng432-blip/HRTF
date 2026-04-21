import trimesh
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
HEADSTL_DIR = Path("dataset/HeadSTL")
OUT_DIR = Path("dataset/HeadNPY")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADSTL_SUFFIX = "_HeadSTL.stl"
OUT_SUFFIX = "_HeadNPY.npy"

SKIP_EXISTING = True   # 如果已存在 .npy 是否跳过


# =========================
# HELPERS
# =========================
def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Trimesh):
        return mesh

    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No valid Trimesh geometry in scene: {path}")
        return trimesh.util.concatenate(geoms)

    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def normalize_point_cloud(x: np.ndarray) -> np.ndarray:
    """
    x: (N,3)
    中心化 + 单位球归一化
    """
    x = x.astype(np.float32)
    x = x - np.mean(x, axis=0, keepdims=True)

    scale = np.max(np.linalg.norm(x, axis=1))
    if scale > 1e-8:
        x = x / scale

    return x.astype(np.float32)


# =========================
# MAIN
# =========================
files = sorted(HEADSTL_DIR.glob(f"P*{HEADSTL_SUFFIX}"))

if not files:
    raise FileNotFoundError(f"No files found in {HEADSTL_DIR} matching *{HEADSTL_SUFFIX}")

print(f"Found {len(files)} HeadSTL files")
print("Convert raw mesh vertices directly to NPY (no sampling)")
print()

ok_cnt = 0
skip_cnt = 0
fail_cnt = 0
stats = []

for f in files:
    pid = f.name.replace(HEADSTL_SUFFIX, "")
    out_path = OUT_DIR / f"{pid}{OUT_SUFFIX}"

    try:
        if SKIP_EXISTING and out_path.exists():
            arr = np.load(out_path)
            print(f"[Skip exist] {f.name} -> {out_path.name} | shape={arr.shape}")
            skip_cnt += 1
            continue

        mesh = load_mesh(f)

        v = len(mesh.vertices)
        face_n = len(mesh.faces)

        # ===== 不采样，直接取原始顶点 =====
        pts = np.asarray(mesh.vertices, dtype=np.float32)   # (N,3)
        pts = normalize_point_cloud(pts)

        np.save(out_path, pts)

        stats.append((f.name, v, face_n, pts.shape[0], out_path.name))
        ok_cnt += 1

        print(f"[OK] {f.name}")
        print(f"     vertices={v}, faces={face_n}")
        print(f"     saved array shape={pts.shape}, dtype={pts.dtype}")
        print(f"     saved to {out_path}")
        print()

    except Exception as e:
        fail_cnt += 1
        print(f"[FAIL] {f.name}: {e}")
        print()

print("\n==============================")
print("Summary")
print("==============================")
print(f"Success: {ok_cnt}")
print(f"Skipped: {skip_cnt}")
print(f"Failed : {fail_cnt}")

if stats:
    verts = np.array([x[1] for x in stats])
    faces = np.array([x[2] for x in stats])
    saved_n = np.array([x[3] for x in stats])

    print("\nProcessed file statistics:")
    print(f"  mean vertices = {verts.mean():.0f}")
    print(f"  mean faces    = {faces.mean():.0f}")
    print(f"  mean saved N  = {saved_n.mean():.0f}")
    print(f"  min saved N   = {saved_n.min()}")
    print(f"  max saved N   = {saved_n.max()}")

print(f"\nAll raw-vertex point clouds are saved in: {OUT_DIR.resolve()}")