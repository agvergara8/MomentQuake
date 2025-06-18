import numpy as np
import pandas as pd
import os

# === CONFIGURACIÓN ===
STEAD_FILE = 'stead_dataset.npz'  # ruta al archivo descargado
OUTPUT_DIR = 'converted_out_files'  # carpeta donde guardar los .out
NUM_SAMPLES = 512     # longitud de cada señal
MAX_FILES = 1000      # máximo de archivos a generar

# === CARGA DEL DATASET ===
print(f'Cargando {STEAD_FILE}...')
data = np.load(STEAD_FILE, allow_pickle=True)["X"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Guardando archivos .out en: {OUTPUT_DIR}')

# === CONVERSIÓN ===
count = 0
for d in data:
    trace = d['trace_data'][:NUM_SAMPLES]
    if len(trace) < NUM_SAMPLES:
        continue  # ignorar señales demasiado cortas

    # Etiqueta binaria
    label = 1 if d['label'] == 'earthquake_local' else 0
    labels = [label] * NUM_SAMPLES

    df = pd.DataFrame({'amplitude': trace, 'label': labels})
    name = d['trace_name'].replace('/', '_').replace('\\', '_')
    filename = f"{name}_{count}.out"
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    count += 1
    if count >= MAX_FILES:
        break

print(f'\n✅ Conversión completada: {count} archivos generados.')
