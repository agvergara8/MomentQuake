import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

# --- Configuraciones ---
DATA_ROOT = "data_stead_raw"  # Carpeta con los archivos .zip descomprimidos
OUTPUT_DIR = "datasets_stead"  # Carpeta de salida para archivos .npy y .csv
WINDOW_SIZE = 512
STRIDE = 512  # Sin solapamiento (stride igual a la longitud de la ventana)
FS = 100  # Frecuencia de muestreo de STEAD (ajustar si es necesario)
START_OFFSET = 0  # Si no se usa el timestamp, este es el desplazamiento temporal
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path="data_stead_raw/chunk3.csv"

def create_windows(data, window_size=512, stride=512):
    """Función para crear las ventanas deslizantes"""
    return np.stack([data[i:i + window_size] for i in range(0, data.shape[0] - window_size + 1, stride)])

def process_file(hdf5_path, csv_path):
    """Procesar cada archivo .hdf5 y .csv"""
    base_name = os.path.splitext(os.path.basename(hdf5_path))[0]

    # Leer datos desde el archivo .hdf5
    with h5py.File(hdf5_path, 'r') as f:
        # Asegúrate de que tienes la clave de los datos
        waveform = f['waveform'][:]  # Datos de la señal sísmica (n_samples, 3)

    # Comprobar si la forma de los datos es la esperada
    if waveform.shape[1] != 3:
        print(f"[ERROR] Datos no esperados en {hdf5_path}")
        return

    # Obtener timestamp inicial (si se tiene en el archivo .csv, o se puede asignar un valor de offset)
    start_time = datetime.utcnow() + timedelta(seconds=START_OFFSET)

    # Crear ventanas deslizantes para cada componente (X, Y, Z)
    for ch_idx, ch_label in enumerate(['X', 'Y', 'Z']):
        ch_data = waveform[:, ch_idx].reshape(-1, 1)  # (n_samples, 1)

        windows = create_windows(ch_data, WINDOW_SIZE, STRIDE)

        # Guardar archivo .npy
        out_base = f"{base_name}_{ch_label}"
        np.save(os.path.join(OUTPUT_DIR, f"{out_base}.npy"), windows)

        # Crear metadatos
        metadata = []
        for i in range(windows.shape[0]):
            timestamp = start_time + timedelta(seconds=i * (STRIDE / FS))
            metadata.append({
                "window_global_idx": i,
                "file": base_name,
                "channel": ch_label,
                "window_in_file": i,
                "timestamp": timestamp.isoformat()
            })

        pd.DataFrame(metadata).to_csv(os.path.join(OUTPUT_DIR, f"{out_base}.csv"), index=False)
        print(f"[INFO] Guardados: {out_base}.npy y .csv")

def main():
    # Buscar los archivos .zip descomprimidos
    hdf5_files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.hdf5')]
    csv_files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.csv')]

    # Asegurarse de que tenemos un .csv por cada archivo .hdf5
    for hdf5_file in tqdm(hdf5_files, desc="Procesando archivos STEAD"):
        base_name = os.path.splitext(hdf5_file)[0]
        csv_file = os.path.join(DATA_ROOT, f"{base_name}.csv")
        if csv_file in csv_files:
            process_file(os.path.join(DATA_ROOT, hdf5_file), csv_file)
        else:
            print(f"[ERROR] No se encontró el archivo CSV correspondiente para {base_name}")

if __name__ == "__main__":
    main()
