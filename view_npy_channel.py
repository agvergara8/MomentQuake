import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# --- Argumentos ---
parser = argparse.ArgumentParser(description="Visualiza un archivo .npy de un canal de geófono")
parser.add_argument("npy_path", type=str, help="Ruta al archivo .npy (ej. datasets/CSIC_LaPalma_Geophone1_X.npy)")
parser.add_argument("--window", type=int, default=0, help="Índice de la ventana a visualizar")
parser.add_argument("--channels", action="store_true", help="Mostrar todas las señales si hay múltiples canales")

args = parser.parse_args()

# --- Cargar datos ---
if not os.path.exists(args.npy_path):
    print(f"[ERROR] Archivo no encontrado: {args.npy_path}")
    exit(1)

data = np.load(args.npy_path)
print(f"[INFO] Shape del archivo: {data.shape}")

if args.window >= data.shape[0]:
    print(f"[ERROR] El índice de ventana es mayor que el total ({data.shape[0]})")
    exit(1)

# --- Visualización ---
plt.figure(figsize=(12, 5))
window = data[args.window]

if window.ndim == 2:
    for ch in range(window.shape[1]):
        plt.plot(window[:, ch], label=f"Canal {ch}", alpha=0.6)
elif window.ndim == 1 or not args.channels:
    plt.plot(window)
    plt.title(f"Ventana {args.window}")
else:
    print("[ERROR] Formato de ventana no reconocido.")
    exit(1)

plt.xlabel("Tiempo")
plt.ylabel("Amplitud")
plt.title(f"Visualización de {os.path.basename(args.npy_path)} - Ventana {args.window}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
