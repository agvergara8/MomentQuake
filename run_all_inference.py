import os
import subprocess

DATASET_DIR = "datasets"

for file in sorted(os.listdir(DATASET_DIR)):
    if file.endswith(".npy"):
        path = os.path.join(DATASET_DIR, file)
        print(f"[INFO] Ejecutando inferencia sobre: {file}")
        subprocess.run(["python", "moment_local_inference.py", "--npy_path", path])
