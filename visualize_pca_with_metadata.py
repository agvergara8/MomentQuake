import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import argparse

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Visualiza PCA de un .npy con su .csv de metadatos")
parser.add_argument("base_path", type=str, help="Ruta base (sin extensión) del archivo .npy y .csv")
args = parser.parse_args()

npy_path = args.base_path + ".npy"
csv_path = args.base_path + ".csv"

# --- Cargar datos ---
data = np.load(npy_path)
meta = pd.read_csv(csv_path)
print(f"[INFO] Datos cargados: {data.shape}, Metadatos: {meta.shape}")

# --- Flatten para PCA ---
N, L, C = data.shape if data.ndim == 3 else (*data.shape, 1)
flat_data = data.reshape(N, L * C)

# --- PCA ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flat_data)

# --- Detección de anomalías ---
detector = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = detector.fit_predict(flat_data)  # -1 = anómalo

# --- Visualización ---
colors = ['red' if a == -1 else 'blue' for a in anomaly_labels]

plt.figure(figsize=(12, 8))
sc = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.7)

# Etiquetar al hacer hover
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

def format_tooltip(index):
    row = meta.iloc[index]
    return (f"Ventana #{index}\n"
            f"{row['timestamp']}\n"
            f"{row['file']}\n"
            f"Canal: {row['source_folder']}")

annot = plt.gca().annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w"),
                           arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    idx = ind["ind"][0]
    pos = sc.get_offsets()[idx]
    annot.xy = pos
    text = format_tooltip(idx)
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor("lightyellow")
    annot.get_bbox_patch().set_alpha(0.9)

def hover(event: MouseEvent):
    vis = annot.get_visible()
    if event.inaxes == plt.gca():
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            plt.draw()
        elif vis:
            annot.set_visible(False)
            plt.draw()

plt.title(f"PCA de {os.path.basename(npy_path)} con anomalías")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.gcf().canvas.mpl_connect("motion_notify_event", hover)
plt.tight_layout()
plt.show()
