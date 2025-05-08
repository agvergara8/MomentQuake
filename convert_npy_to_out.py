import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# === Configuración ===
INPUT_DIR = "datasets"  # Carpeta donde están los .npy
OUTPUT_DIR = "data/TimeseriesDatasets/anomaly_detection/CSIC_LaPalma"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLES_PER_FILE = 15000
START_TIME = datetime.strptime("00:00:32", "%H:%M:%S")  # inicio del primer archivo
SAMPLING_RATE = 250  # Hz
DURATION = timedelta(seconds=SAMPLES_PER_FILE / SAMPLING_RATE)
FIXED_DATE = "2025-01-01"  # puedes cambiarlo si quieres adaptar por fecha real

# === Conversión ===
for filename in sorted(os.listdir(INPUT_DIR)):
    if filename.endswith(".npy") and "CSIC_LaPalma_Geophone" in filename:
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"Procesando {filename}...")
        
        # Extraer geófono y canal
        parts = filename.replace(".npy", "").split("_")
        geophone = parts[2].replace("Geophone", "")
        channel = parts[3].upper()

        # Cargar datos
        data = np.load(filepath)
        num_blocks = len(data) // SAMPLES_PER_FILE

        for i in range(num_blocks):
            segment = data[i*SAMPLES_PER_FILE : (i+1)*SAMPLES_PER_FILE]
            segment_df = pd.DataFrame({
                "value": segment,
                "label": 0
            })

            # Calcular timestamp
            block_time = START_TIME + i * DURATION
            time_str = block_time.strftime("%H%M%S")
            out_name = f"Geo{geophone}_{channel}_{FIXED_DATE}_{time_str}.out"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            segment_df.to_csv(out_path, index=False)

print("✅ Conversión completa.")
