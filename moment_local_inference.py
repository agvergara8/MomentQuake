import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import os
import yaml
from types import SimpleNamespace
import sys
from transformers import T5Config
import argparse

# --- Argumentos desde línea de comandos ---
parser = argparse.ArgumentParser()
parser.add_argument("--npy_path", type=str, default="datasets/CSIC_LaPalma_Geophone1_X.npy", help="Ruta al archivo .npy")
args = parser.parse_args()

# --- Añadir ruta local del repositorio momentfm ---
sys.path.append(os.path.join(os.path.dirname(__file__), "momentfm"))
from models.moment import MOMENT

# --- Configuración general ---
BLOCK_SIZE = 10000

# --- Cargar config YAML y convertirlo en Namespace ---
with open("configs/moment_base.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = SimpleNamespace(**config_dict)
model_config = T5Config.from_dict(config.t5_config)

# --- Inicializar modelo ---
model = MOMENT(config)
model.eval()

# --- Cargar dataset completo (sin normalizar) ---
data = np.load(args.npy_path)
print(f"[INFO] Dataset cargado: {data.shape}")

# --- Crear carpeta de salida para ventanas detectadas ---
output_dir = os.path.join("detected_windows", os.path.splitext(os.path.basename(args.npy_path))[0])
os.makedirs(output_dir, exist_ok=True)

# --- Procesar por bloques ---
total_windows = data.shape[0]
saved = 0
for start in range(0, total_windows, BLOCK_SIZE):
    end = min(start + BLOCK_SIZE, total_windows)
    print(f"[INFO] Procesando ventanas {start} a {end}")

    # --- Seleccionar bloque y convertir a tensor ---
    block = data[start:end]
    block_tensor = torch.tensor(block, dtype=torch.float32)

    if block_tensor.ndim == 2:
        block_tensor = block_tensor.unsqueeze(-1)

    block_tensor = block_tensor.permute(0, 2, 1)  # (N, C, L)
    
    # --- Normalizar bloque ---
    mean = block_tensor.mean(dim=(0, 1), keepdim=True)
    std = block_tensor.std(dim=(0, 1), keepdim=True) + 1e-6
    block_norm = (block_tensor - mean) / std
    
    # --- Pasar por Moment ---
    with torch.no_grad():
        outputs = model(x_enc=block_norm)
    recon = outputs.reconstruction.cpu().numpy()

    # --- Flatten y PCA/IF ---
    features = recon.reshape(recon.shape[0], -1)
    detector = IsolationForest(contamination=0.01, random_state=42)
    anomaly_scores = detector.fit_predict(features)

    # --- Guardar ventanas detectadas ---
    event_indices = np.where(anomaly_scores == -1)[0]
    print(f"[INFO] Se detectaron {len(event_indices)} eventos en este bloque")

    for i, idx in enumerate(event_indices):
        global_idx = start + idx
        out_path = os.path.join(output_dir, f"window_{global_idx:05d}.npy")
        np.save(out_path, data[global_idx])
        saved += 1
        


print(f"[INFO] Proceso completado. Total de eventos guardados: {saved}")

