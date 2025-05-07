import numpy as np
import torch
from momentfm.models.moment import MOMENT
import yaml
from types import SimpleNamespace
import os
import pandas as pd

# --- Parámetros de entrada ---
DATA_PATH = "datasets/CSIC_LaPalma_Geophone1_X.npy"  # Ruta a tu archivo .npy
CONFIG_PATH = "configs/moment_base.yaml"             # Configuración del modelo
UMBRAL_PERCENTIL = 95                                 # Umbral de clasificación (percentil del error)
BLOCK_SIZE = 5000  # Número de ventanas a procesar por bloque (aumentado para menos interacción)

# --- Preguntar al usuario si quiere procesar todo o solo parte ---
print("\n[INTERACTIVO] Elige el modo de ejecución:")
print("1 - Procesar TODAS las ventanas")
print("2 - Procesar por bloques con confirmación")
modo = input("Introduce 1 o 2: ").strip()
if modo not in ["1", "2"]:
    print("[ERROR] Modo no válido. Saliendo.")
    exit(1)

# --- Cargar configuración y modelo ---
with open(CONFIG_PATH, "r") as f:
    config_dict = yaml.safe_load(f)
config_dict["task_name"] = "reconstruction"
config = SimpleNamespace(**config_dict)

model = MOMENT(config)
model.eval()

# --- Cargar datos ---
print(f"[INFO] Cargando datos desde {DATA_PATH}")
data = np.load(DATA_PATH)
if data.ndim == 3 and data.shape[1] != 512:
    data = np.transpose(data, (0, 2, 1))  # (N, 512, C)

N = data.shape[0]
print(f"[INFO] Total de ventanas: {N}")

# --- Procesar en bloques ---
all_mse = []
for start in range(0, N, BLOCK_SIZE):
    end = min(start + BLOCK_SIZE, N)
    print(f"[INFO] Procesando ventanas {start} a {end-1}")

    data_block = data[start:end]
    data_tensor = torch.tensor(data_block, dtype=torch.float32).permute(0, 2, 1)  # (B, C, 512)

    # Normalización local por bloque
    mean = data_tensor.mean(dim=(0, 2), keepdim=True)
    std = data_tensor.std(dim=(0, 2), keepdim=True) + 1e-6
    data_norm = (data_tensor - mean) / std

    with torch.no_grad():
        output = model.forward(x_enc=data_norm)
        reconstruction = output.reconstruction

    mse = ((data_norm - reconstruction) ** 2).mean(dim=(1, 2)).cpu().numpy()
    all_mse.append(mse)

    if modo == "2":
        cont = input("\n[INTERACTIVO] ¿Procesar siguiente bloque? (s/n): ").strip().lower()
        if cont != "s":
            print("[INFO] Proceso detenido por el usuario.")
            break

# --- Concatenar errores y clasificar ---
all_mse = np.concatenate(all_mse)
threshold = np.percentile(all_mse, UMBRAL_PERCENTIL)
event_mask = all_mse > threshold

print(f"[INFO] Umbral = {threshold:.4f} (percentil {UMBRAL_PERCENTIL})")
print(f"[INFO] Eventos detectados: {np.sum(event_mask)} / {len(all_mse)}")

# --- Guardar resultados numpy ---
np.save("reconstruction_errors.npy", all_mse)
np.save("predicted_event_mask.npy", event_mask.astype(np.uint8))
print("[INFO] Guardado: reconstruction_errors.npy y predicted_event_mask.npy")

# --- Enriquecer CSV con resultados ---
csv_path = DATA_PATH.replace(".npy", ".csv")
if os.path.exists(csv_path):
    print(f"[INFO] Enriqueciendo archivo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(all_mse) > len(df):
        print("[ERROR] Más errores calculados que ventanas en el CSV. Algo está mal.")
        exit(1)
    df_partial = df.iloc[:len(all_mse)].copy()
    df_partial["reconstruction_error"] = all_mse
    df_partial["es_evento"] = event_mask.astype(int)
    output_csv = csv_path.replace(".csv", "_resultados.csv")
    df_partial.to_csv(output_csv, index=False)
    print(f"[INFO] Guardado CSV enriquecido: {output_csv}")
else:
    print(f"[ADVERTENCIA] No se encontró CSV: {csv_path}. No se generó resumen tabular.")
