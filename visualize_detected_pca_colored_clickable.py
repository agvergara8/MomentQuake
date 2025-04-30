import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

# --- Recoger todas las ventanas ---
root_dir = "detected_windows"
windows = []
labels = []
colors_map = {}
color_palette = plt.cm.get_cmap("tab20")

for i, folder in enumerate(sorted(os.listdir(root_dir))):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".npy"):
            full_path = os.path.join(folder_path, file)
            try:
                win = np.load(full_path).squeeze()
                windows.append(win)
                labels.append({
                    "canal": folder,
                    "archivo": file,
                    "path": full_path
                })
                colors_map[folder] = color_palette(i % 20)
            except Exception as e:
                print(f"[WARNING] Error cargando {full_path}: {e}")

print(f"[INFO] Ventanas cargadas: {len(windows)}")

# --- PCA ---
data = np.stack(windows)
data_flat = data.reshape(data.shape[0], -1)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_flat)

# --- Colores por canal ---
colors = [colors_map[l["canal"]] for l in labels]

# --- Visualización interactiva ---
fig, ax = plt.subplots(figsize=(12, 8))
sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6)

annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def format_tooltip(index):
    meta = labels[index]
    return f"{meta['canal']}\n{meta['archivo']}"

def update_annot(ind):
    idx = ind["ind"][0]
    pos = sc.get_offsets()[idx]
    annot.xy = pos
    annot.set_text(format_tooltip(idx))
    annot.get_bbox_patch().set_facecolor("lightyellow")
    annot.get_bbox_patch().set_alpha(0.9)

def hover(event: MouseEvent):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

def on_click(event: MouseEvent):
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            idx = ind["ind"][0]
            meta = labels[idx]
            signal = np.load(meta["path"]).squeeze()
            plt.figure(figsize=(10, 3))
            plt.plot(signal)
            plt.title(f"{meta['canal']} - {meta['archivo']}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect("button_press_event", on_click)

ax.set_title("PCA de ventanas detectadas (color por canal)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.grid(True)
plt.tight_layout()
plt.show()
