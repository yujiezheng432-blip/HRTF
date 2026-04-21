import trimesh
import numpy as np
from pathlib import Path

STL_DIR = Path("dataset/stl")

MAX_FILES = 10   # 只读取前10个

vertex_counts = []
face_counts = []

files = sorted(STL_DIR.glob("P*.stl"))[:MAX_FILES]

print("Checking first", len(files), "STL files\n")

for f in files:
    mesh = trimesh.load(f, process=False)

    v = len(mesh.vertices)
    fcount = len(mesh.faces)

    vertex_counts.append(v)
    face_counts.append(fcount)

    print(f"{f.name}: vertices={v}, faces={fcount}")

vertex_counts = np.array(vertex_counts)
face_counts = np.array(face_counts)

print("\n============================")
print("Statistics (first 10 files)")
print("============================")

print("Vertex count:")
print("min   :", vertex_counts.min())
print("max   :", vertex_counts.max())
print("mean  :", int(vertex_counts.mean()))
print("median:", int(np.median(vertex_counts)))

print()

print("Face count:")
print("min   :", face_counts.min())
print("max   :", face_counts.max())
print("mean  :", int(face_counts.mean()))

print("\n============================")

mean_v = int(vertex_counts.mean())

print("Estimated mesh density:", mean_v, "vertices")

if mean_v > 1_000_000:
    print("Observation: Mesh is extremely dense (millions of vertices)")
    print("Suggestion: Random sampling to 5000–20000 points is sufficient for ML")
elif mean_v > 10000:
    print("Mesh density is high, sampling to ~10k points recommended")
else:
    print("Mesh density is low, consider increasing point count")
