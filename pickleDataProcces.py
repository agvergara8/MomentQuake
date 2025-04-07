import os
import pickle
import numpy as np
from obspy.core.stream import Stream

# --- Configuraciones ---
DATA_ROOT = "dataPickle/CSIC_LaPalma_Geophone"
GEOPHONES = range(1, 9)
CHANNELS = ["X", "Y", "Z"]
WINDOW_SIZE = 512
STRIDE = 256

# --- Función para cargar un archivo pickle como array ---
def load_stream_as_array(path):
    try:
        with open(path, "rb") as f:
            stream = pickle.load(f)
            print(f"[DEBUG] Tipo cargado en {path}: {type(stream)}")
            if not isinstance(stream, Stream):
                return None
            return np.stack([tr.data for tr in stream], axis=1)  # shape: (timesteps, channels)
    except Exception as e:
        print(f"[WARNING] Error loading {path}: {e}")
        return None

# --- Función para crear ventanas deslizantes ---
def create_windows(data, window_size=512, stride=256):
    windows = []
    for i in range(0, data.shape[0] - window_size + 1, stride):
        windows.append(data[i:i+window_size])
    return np.stack(windows)

# --- Recorrer toda la estructura de carpetas ---
def collect_all_data():
    all_windows = []

    for geophone in GEOPHONES:
        for channel in CHANNELS:
            folder_path = f"{DATA_ROOT}{geophone}_{channel}"
            print(f"[DEBUG] Buscando en {folder_path}")
            if not os.path.exists(folder_path):
                print(f"[DEBUG] Carpeta no existe: {folder_path}")
                continue

            for filename in os.listdir(folder_path):
                if filename.endswith(".pickle"):
                    file_path = os.path.join(folder_path, filename)
                    print(f"[DEBUG] Cargando archivo: {file_path}")
                    arr = load_stream_as_array(file_path)
                    if arr is None:
                        continue

                    channel_index = (geophone - 1) * 3 + CHANNELS.index(channel)

                    if len(all_windows) <= channel_index:
                        all_windows.extend([[] for _ in range(channel_index - len(all_windows) + 1)])

                    all_windows[channel_index].append(arr)

    streams = []
    for i, stream_list in enumerate(all_windows):
        if stream_list:
            print(f"[DEBUG] Canal {i}: {len(stream_list)} fragmentos")
            streams.append(np.concatenate(stream_list, axis=0))
        else:
            print(f"[DEBUG] Canal {i}: sin datos")

    if not streams:
        raise ValueError("No se encontraron datos válidos.")

    full_data = np.stack(streams, axis=1)  # shape: (timesteps, total_channels)

    print(f"[INFO] Datos combinados: {full_data.shape}")

    windows = create_windows(full_data, WINDOW_SIZE, STRIDE)
    print(f"[INFO] Total de ventanas creadas: {windows.shape}")

    return windows

if __name__ == "__main__":
    dataset = collect_all_data()
    np.save("moment_ready_dataset.npy", dataset)
    print("[INFO] Dataset guardado como 'moment_ready_dataset.npy'")
