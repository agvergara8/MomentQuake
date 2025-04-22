import os
import pickle
import numpy as np
from obspy.core.stream import Stream
from tqdm import tqdm

# --- Configuraciones ---
DATA_ROOT = "dataPickle/CSIC_LaPalma_Geophone"
DATASETS_DIR = "datasets"
GEOPHONES = range(1, 9)
CHANNELS = ["X", "Y", "Z"]
WINDOW_SIZE = 512
STRIDE = 256

os.makedirs(DATASETS_DIR, exist_ok=True)

# --- Crear ventanas deslizantes ---
def create_windows(data, window_size=512, stride=256):
    return np.stack([data[i:i+window_size] for i in range(0, data.shape[0] - window_size + 1, stride)])

# --- Cargar .pickle a array ---
def load_stream_as_array(path):
    try:
        with open(path, "rb") as f:
            stream = pickle.load(f)
            if not isinstance(stream, Stream):
                return None
            return np.stack([tr.data for tr in stream], axis=1)
    except Exception as e:
        print(f"[WARNING] Error loading {path}: {e}")
        return None

# --- Procesamiento por carpeta ---
def process_folder(folder_path):
    all_segments = []

    for filename in tqdm(os.listdir(folder_path), desc=f"Procesando {os.path.basename(folder_path)}"):
        if not filename.endswith(".pickle"):
            continue

        file_path = os.path.join(folder_path, filename)
        arr = load_stream_as_array(file_path)
        if arr is None:
            continue

        all_segments.append(arr)

    if not all_segments:
        print(f"[INFO] No se encontraron datos válidos en {folder_path}")
        return

    full_array = np.concatenate(all_segments, axis=0)
    print(f"[INFO] Datos combinados: {full_array.shape} en {folder_path}")

    windows = create_windows(full_array, WINDOW_SIZE, STRIDE)
    print(f"[INFO] Ventanas creadas: {windows.shape} en {folder_path}")

    out_name = os.path.basename(folder_path) + ".npy"
    out_path = os.path.join(DATASETS_DIR, out_name)
    np.save(out_path, windows)
    print(f"[INFO] Guardado: {out_path}")

if __name__ == "__main__":
    for geophone in GEOPHONES:
        for channel in CHANNELS:
            folder_path = f"{DATA_ROOT}{geophone}_{channel}"
            if os.path.exists(folder_path):
                process_folder(folder_path)
            else:
                print(f"[INFO] Carpeta no encontrada: {folder_path}")
