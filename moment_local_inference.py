import numpy as np
import torch
from momentfm.models.moment import MOMENT
import yaml
from types import SimpleNamespace
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Parámetros de entrada ---
CONFIG_PATH = "configs/moment_base.yaml"             # Configuración del modelo
UMBRAL_PERCENTIL = 95                                 # Umbral de clasificación (percentil del error)
BLOCK_SIZE = 5000                                     # Tamaño de bloque
INPUT_FOLDER = "datasets"
OUTPUT_FOLDER = "resultados_inferencia"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Preguntar al usuario si quiere procesar todo o por bloques ---
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

# --- Procesar todos los archivos .npy en el input folder ---
npy_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".npy")]

for file_name in npy_files:
    data_path = os.path.join(INPUT_FOLDER, file_name)
    base_name = file_name.replace(".npy", "")
    print(f"\n[INFO] Procesando archivo: {file_name}")

    data = np.load(data_path)
    if data.ndim == 3 and data.shape[1] != 512:
        data = np.transpose(data, (0, 2, 1))  # (N, 512, C)

    N = data.shape[0]
    print(f"[INFO] Total de ventanas: {N}")

    all_mse = []
    for start in tqdm(range(0, N, BLOCK_SIZE), desc=f"Procesando {file_name}"):
        end = min(start + BLOCK_SIZE, N)
        data_block = data[start:end]
        data_tensor = torch.tensor(data_block, dtype=torch.float32).permute(0, 2, 1)

        mean = data_tensor.mean(dim=(0, 2), keepdim=True)
        std = data_tensor.std(dim=(0, 2), keepdim=True) + 1e-6
        data_norm = (data_tensor - mean) / std

        with torch.no_grad():
            output = model.forward(x_enc=data_norm)
            reconstruction = output.reconstruction

        mse = ((data_norm - reconstruction) ** 2).mean(dim=(1, 2)).cpu().numpy()
        all_mse.append(mse)

        if modo == "2":
            cont = input("\n¿Procesar siguiente bloque? (s/n): ").strip().lower()
            if cont != "s":
                print("[INFO] Proceso detenido por el usuario.")
                break

    all_mse = np.concatenate(all_mse)
    threshold = np.percentile(all_mse, UMBRAL_PERCENTIL)
    event_mask = all_mse > threshold

    print(f"[INFO] Umbral = {threshold:.4f} (percentil {UMBRAL_PERCENTIL})")
    print(f"[INFO] Eventos detectados: {np.sum(event_mask)} / {len(all_mse)}")

    # Guardar errores y máscara
    np.save(os.path.join(OUTPUT_FOLDER, f"{base_name}_errors.npy"), all_mse)
    np.save(os.path.join(OUTPUT_FOLDER, f"{base_name}_event_mask.npy"), event_mask.astype(np.uint8))

    # Enriquecer CSV si existe
    csv_path = os.path.join(INPUT_FOLDER, base_name + ".csv")
    if os.path.exists(csv_path):
        print(f"[INFO] Enriqueciendo CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if len(all_mse) > len(df):
            print("[ERROR] Más errores que ventanas. Saltando este CSV.")
            continue
        df_partial = df.iloc[:len(all_mse)].copy()
        df_partial["reconstruction_error"] = all_mse
        df_partial["es_evento"] = event_mask.astype(int)

        output_csv = os.path.join(OUTPUT_FOLDER, f"{base_name}_resultados.csv")
        df_partial.to_csv(output_csv, index=False)
        print(f"[INFO] Guardado CSV enriquecido: {output_csv}")

        # Visualización básica opcional
        try:
            df_partial["timestamp"] = pd.to_datetime(df_partial["timestamp"].astype(str).str.replace(",", "."), errors="coerce")
            df_partial = df_partial[df_partial["timestamp"].notnull()].copy()
            df_partial = df_partial.sort_values("timestamp")

            plt.figure(figsize=(14, 5))
            plt.plot(df_partial["timestamp"], df_partial["reconstruction_error"], label="Error de reconstrucción (MSE)", color="blue")
            plt.scatter(
                df_partial[df_partial["es_evento"] == 1]["timestamp"],
                df_partial[df_partial["es_evento"] == 1]["reconstruction_error"],
                color="red", label="Evento detectado", s=20
            )
            plt.title(f"Eventos detectados: {base_name}")
            plt.xlabel("Tiempo")
            plt.ylabel("Error de reconstrucción")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"{base_name}_grafico.png"))
            plt.close()
        except Exception as e:
            print(f"[AVISO] Visualización fallida: {e}")
    else:
        print(f"[ADVERTENCIA] No se encontró CSV para: {base_name}")
