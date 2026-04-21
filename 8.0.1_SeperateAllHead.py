import trimesh
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
STL_DIR = Path("dataset/stl")
OUT_DIR = Path("dataset/HeadSTL")
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_THRESHOLD = -100.0   # 保留 x > -100 的部分


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
            raise ValueError("Scene contains no valid Trimesh geometry.")
        return trimesh.util.concatenate(geoms)

    raise TypeError(f"Unsupported mesh type: {type(mesh)}")


def crop_head_region(mesh: trimesh.Trimesh, x_threshold: float) -> trimesh.Trimesh:
    """
    保留 x > x_threshold 的头部区域。
    规则：如果一个 face 的任意顶点满足 x > x_threshold，就保留该 face。
    """
    vertices = mesh.vertices
    faces = mesh.faces

    face_vertices = vertices[faces]   # (F, 3, 3)
    face_x = face_vertices[:, :, 0]   # (F, 3)

    keep_faces = np.any(face_x > x_threshold, axis=1)

    if np.sum(keep_faces) == 0:
        raise ValueError(f"No faces found with x > {x_threshold}")

    cropped = mesh.submesh([keep_faces], append=True, repair=True)
    return cropped


# =========================
# MAIN
# =========================
files = sorted(STL_DIR.glob("P*.stl"))

if not files:
    raise FileNotFoundError(f"No STL files found in {STL_DIR}")

print(f"Processing all STL files: {len(files)} files\n")

stats = []
ok_cnt = 0
fail_cnt = 0

for f in files:
    try:
        mesh = load_mesh(f)

        v0 = len(mesh.vertices)
        f0 = len(mesh.faces)

        head_mesh = crop_head_region(mesh, X_THRESHOLD)

        v1 = len(head_mesh.vertices)
        f1 = len(head_mesh.faces)

        out_path = OUT_DIR / f"{f.stem}_HeadSTL.stl"
        head_mesh.export(out_path)

        stats.append((f.name, v0, f0, v1, f1, out_path.name))
        ok_cnt += 1

        print(f"[OK] {f.name}")
        print(f"     original : vertices={v0}, faces={f0}")
        print(f"     cropped  : vertices={v1}, faces={f1}")
        print(f"     saved to : {out_path}")
        print()

    except Exception as e:
        fail_cnt += 1
        print(f"[FAIL] {f.name}: {e}")
        print()

print("\n==============================")
print("Summary")
print("==============================")
print(f"Success: {ok_cnt}")
print(f"Failed : {fail_cnt}")

for name, v0, f0, v1, f1, out_name in stats:
    print(f"{name} -> {out_name}")
    print(f"  original: vertices={v0}, faces={f0}")
    print(f"  cropped : vertices={v1}, faces={f1}")

if stats:
    orig_v = np.array([x[1] for x in stats])
    crop_v = np.array([x[3] for x in stats])
    orig_f = np.array([x[2] for x in stats])
    crop_f = np.array([x[4] for x in stats])

    print("\nAverage statistics (all files):")
    print(f"  original vertices mean = {orig_v.mean():.0f}")
    print(f"  cropped  vertices mean = {crop_v.mean():.0f}")
    print(f"  original faces mean    = {orig_f.mean():.0f}")
    print(f"  cropped  faces mean    = {crop_f.mean():.0f}")

print(f"\nAll cropped head STL files are saved in: {OUT_DIR.resolve()}")