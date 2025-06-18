import os
import pickle
import numpy as np
import pandas as pd
from obspy.core.stream import Stream
from tqdm import tqdm
from datetime import datetime, timedelta

# --- Configuraciones ---
DATA_ROOT = "dataPickle/CSIC_LaPalma_Geophone"
DATASETS_DIR = "datasets"
GEOPHONES = range(1, 9)
CHANNELS = ["X", "Y", "Z"]
WINDOW_SIZE = 512
STRIDE = 256
FS = 250  # Frecuencia de muestreo en Hz
START_OFFSET = 32  # segundos desde medianoche

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

# --- Procesar una carpeta y generar .npy + metadatos ---
def process_folder(folder_path):
    all_segments = []
    metadata = []

    folder_name = os.path.basename(folder_path)

    for filename in tqdm(os.listdir(folder_path), desc=f"Procesando {folder_name}"):
        if not filename.endswith(".pickle"):
            continue

        file_path = os.path.join(folder_path, filename)
        arr = load_stream_as_array(file_path)
        if arr is None:
            continue

        start_time = datetime.strptime(filename.split(" at ")[0], "%d-%b-%Y") + timedelta(seconds=START_OFFSET)
        windows = create_windows(arr, WINDOW_SIZE, STRIDE)

        all_segments.append(windows)

        for i in range(windows.shape[0]):
            timestamp = start_time + timedelta(seconds=i * (STRIDE / FS))
            metadata.append({
                "window_global_idx": len(metadata),
                "file": filename,
                "window_in_file": i,
                "timestamp": timestamp.isoformat(),
                "source_folder": folder_name
            })

    if not all_segments:
        print(f"[INFO] No se encontraron datos válidos en {folder_path}")
        return

    full_array = np.concatenate(all_segments, axis=0)
    print(f"[INFO] Ventanas combinadas: {full_array.shape} en {folder_path}")

    out_base = os.path.join(DATASETS_DIR, folder_name)
    np.save(out_base + ".npy", full_array)
    pd.DataFrame(metadata).to_csv(out_base + ".csv", index=False)
    print(f"[INFO] Guardados: {out_base}.npy y .csv")

if __name__ == "__main__":
    for geophone in GEOPHONES:
        for channel in CHANNELS:
            folder_path = f"{DATA_ROOT}{geophone}_{channel}"
            if os.path.exists(folder_path):
                process_folder(folder_path)
            else:
                print(f"[INFO] Carpeta no encontrada: {folder_path}")