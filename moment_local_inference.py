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


# --- Añadir ruta local del repositorio momentfm ---
sys.path.append(os.path.join(os.path.dirname(__file__), "momentfm"))

# --- Importar clase MOMENT directamente ---
from models.moment import MOMENT

# --- Cargar dataset ---
data = np.load("moment_ready_dataset.npy")  # shape: (N, 512, 24)
print(f"[INFO] Dataset cargado: {data.shape}")

# --- Normalización simple por canal ---
data_tensor = torch.tensor(data, dtype=torch.float32)
# Reordenar: (batch, seq_len, features, 1) -> (batch, features, seq_len)
data_tensor = data_tensor.squeeze(-1).permute(0, 2, 1)
mean = data_tensor.mean(dim=(0, 1), keepdim=True)  # media por canal
std = data_tensor.std(dim=(0, 1), keepdim=True) + 1e-6  # std por canal

print("[INFO] Normalizando datos...")
data_norm = (data_tensor - mean) / std

# --- Cargar config YAML y convertirlo en Namespace ---
with open("configs/moment_base.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = SimpleNamespace(**config_dict)
model_config = T5Config.from_dict(config.t5_config)
# --- Inicializar modelo ---
model = MOMENT(config)
model.eval()

# --- Obtener embeddings desde el CLS token ---
with torch.no_grad():
    outputs = model(x_enc=data_norm)
    # --- Obtener reconstrucción desde el modelo ---
reconstructed = outputs.reconstruction  # shape: (N, C, L)

# --- Flatten para análisis (por ventana) ---
# Ejemplo: usar reconstrucción como feature vector
recon_np = reconstructed.cpu().numpy()
N, C, L = recon_np.shape
features = recon_np.reshape(N, C * L)  # shape: (N, C×L)

print(f"[INFO] Reconstrucciones generadas: {features.shape}")

# --- Visualización con PCA ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title("Proyección PCA de Reconstrucciones de Moment")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.savefig("pca_moment_reconstruction_local.png")
plt.show()

# --- Detección de anomalías con Isolation Forest ---
detector = IsolationForest(contamination=0.05, random_state=42)
anomaly_scores = detector.fit_predict(features)

event_indices = np.where(anomaly_scores == -1)[0]
print(f"[INFO] Se detectaron {len(event_indices)} posibles eventos sísmicos")

# --- Guardar ventanas detectadas ---
events = data[event_indices]
np.save("detected_events_windows.npy", events)
print("[INFO] Ventanas con eventos guardadas en 'detected_events_windows.npy'")
