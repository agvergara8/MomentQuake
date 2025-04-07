import numpy as np
import matplotlib.pyplot as plt
import argparse

# --- Argumentos desde línea de comandos ---
parser = argparse.ArgumentParser()
parser.add_argument("npy_path", type=str, help="Ruta al archivo .npy")
parser.add_argument("--window", type=int, default=0, help="Índice de la ventana a visualizar")
args = parser.parse_args()

# --- Cargar datos ---
data = np.load(args.npy_path)
print(f"[INFO] Shape del archivo: {data.shape}")

# --- Comprobar dimensiones ---
if data.ndim != 4:
    raise ValueError("Este visualizador espera datos con shape (N, C, L)")

N, C, L, J = data.shape
w = args.window
if w >= N:
    raise IndexError(f"Índice de ventana fuera de rango (máximo {N-1})")

data = np.load("detected_events_windows.npy")
print(f"Shape del array: {data.shape}")

n_ventanas = data.shape[0]
print(f"Número de ventanas: {n_ventanas}")

# --- Visualización ---
plt.figure(figsize=(14, 6))
for ch in range(C):
    plt.plot(data[w, ch], label=f"Canal {ch}", alpha=0.6)

plt.title(f"Ventana {w} - Señales multicanal")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
