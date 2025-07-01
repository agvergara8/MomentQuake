import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import gc  # Para liberar memoria

# --- Configuración ---
DATA_ROOT = "data_stead_raw"
OUTPUT_DIR = "datasets_stead_blocks"
WINDOW_SIZE = 512
STRIDE = 512
FS = 100  # Hz
TARGET_BLOCK_SIZE_GB = 2  # Puedes cambiar esto fácilmente

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_windows(data, window_size=512, stride=512):
    return np.stack([data[i:i + window_size] for i in range(0, data.shape[0] - window_size + 1, stride)])

def estimate_size_gb(array_list):
    total_elements = sum(a.size for a in array_list)
    return (total_elements * 4) / 1e9  # float32 -> 4 bytes

def save_and_clear(buffer_data, buffer_meta, block_idx):
    # Guardar datos y CSV
    block_array = np.concatenate(buffer_data, axis=0)
    out_npy = os.path.join(OUTPUT_DIR, f"dataset_block_{block_idx}.npy")
    out_csv = os.path.join(OUTPUT_DIR, f"dataset_block_{block_idx}.csv")
    np.save(out_npy, block_array)
    pd.DataFrame(buffer_meta).to_csv(out_csv, index=False)
    print(f"[INFO] Guardados: {out_npy}, {out_csv} (≈{estimate_size_gb(buffer_data):.2f} GB)")

    # Liberar memoria
    del block_array
    del buffer_data[:]
    del buffer_meta[:]
    gc.collect()

def main():
    hdf5_path = os.path.join(DATA_ROOT, "merged.hdf5")
    csv_path = os.path.join(DATA_ROOT, "merged.csv")

    if not os.path.exists(hdf5_path) or not os.path.exists(csv_path):
        print("[ERROR] merged.hdf5 o merged.csv no encontrados.")
        return

    meta_df = pd.read_csv(csv_path)
    block_idx = 0
    buffer_data = []
    buffer_meta = []

    with h5py.File(hdf5_path, "r") as f:
        for trace_name in tqdm(f['data'].keys(), desc="Procesando trazas"):
            dataset = f['data'][trace_name]
            data = np.array(dataset)  # (n_samples, 3)

            if data.shape[1] != 3:
                print(f"[WARNING] Formato inesperado en {trace_name}, se omite.")
                continue

            meta_row = meta_df[meta_df['trace_name'] == trace_name]
            if meta_row.empty:
                continue

            network = dataset.attrs.get('network_code', 'NA')
            receiver = dataset.attrs.get('receiver_code', 'NA')
            station_code = f"{network}.{receiver}"

            for idx, label in enumerate(['E', 'N', 'Z']):
                ch_data = data[:, idx].reshape(-1, 1)
                windows = create_windows(ch_data, WINDOW_SIZE, STRIDE)

                buffer_data.append(windows)
                for i in range(windows.shape[0]):
                    buffer_meta.append({
                        "window_global_idx": i,
                        "trace_name": trace_name,
                        "station_code": station_code,
                        "channel": label,
                        "window_in_file": i,
                        "timestamp": datetime.utcnow().isoformat(),
                        "p_arrival_sample": dataset.attrs.get("p_arrival_sample", -1),
                        "s_arrival_sample": dataset.attrs.get("s_arrival_sample", -1),
                        "coda_end_sample": dataset.attrs.get("coda_end_sample", -1)
                    })

                size_gb = estimate_size_gb(buffer_data)
                if size_gb >= TARGET_BLOCK_SIZE_GB:
                    save_and_clear(buffer_data, buffer_meta, block_idx)
                    block_idx += 1

    # Guardar el último bloque si queda algo
    if buffer_data:
        save_and_clear(buffer_data, buffer_meta, block_idx)

if __name__ == "__main__":
    main()
