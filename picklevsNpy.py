import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from obspy.core.stream import Stream

# --- Configuración ---
DATASETS_DIR = "datasets"
PICKLE_ROOT = "dataPickle"
GEOPHONE = "Geophone1"
CHANNEL = "X"
STRIDE = 512
WINDOW_SIZE = 512

# --- Input ---
idx = int(input("Introduce el window_global_idx que quieres comparar: "))
base = f"CSIC_LaPalma_{GEOPHONE}_{CHANNEL}"

# --- Cargar CSV de metadatos ---
csv_path = os.path.join(DATASETS_DIR, f"{base}.csv")
df = pd.read_csv(csv_path)


row = df[df["window_global_idx"] == idx]
if row.empty:
    print(f"No se encontró la ventana {idx} en el CSV.")
    exit()
    
row = row.iloc[0]
window_in_file = row["window_in_file"]
filename = row["file"]
filename = "13-Feb-2025 at 00.00.32.pickle"
pickle_folder = os.path.join(PICKLE_ROOT, f"CSIC_LaPalma_{GEOPHONE}_{CHANNEL}")
pickle_path = os.path.join(pickle_folder, filename)

# --- Cargar .npy y extraer ventana ---
npy_path = os.path.join(DATASETS_DIR, f"{base}.npy")
data = np.load(npy_path)
if data.ndim == 3:
    ventana_npy = data[idx, :, 0]  # canal X
else:
    ventana_npy = data[idx]


print(f"[Path pickle: {pickle_path}")
print(f"[Path npy: {npy_path}")

# --- Cargar .pickle y extraer ventana ---
with open(pickle_path, "rb") as f:
    stream = pickle.load(f)
    if not isinstance(stream, Stream):
        print(f"Archivo no es Stream: {pickle_path}")
        exit()
    arr = np.stack([tr.data for tr in stream], axis=1)
    ventana_pickle = arr[window_in_file * STRIDE : window_in_file * STRIDE + WINDOW_SIZE, 0]

# --- Visualización ---
plt.figure(figsize=(12, 4))
plt.plot(ventana_npy, label="Ventana desde .npy", linestyle='--')
plt.plot(ventana_pickle, label="Ventana desde .pickle", alpha=0.7)
plt.title(f"Comparación de ventana {idx}")
plt.xlabel("Muestra")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
