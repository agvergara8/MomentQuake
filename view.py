import numpy as np
import matplotlib.pyplot as plt

# --- Configuración ---
ARCHIVO = "datasets/CSIC_LaPalma_Geophone1_X.npy"
NUM_VENTANAS = 5
sample_rate = 250  # Hz

# --- Cargar datos ---
data = np.load(ARCHIVO)
if data.ndim == 3:
    data = data[:, :, 0]  # Usa solo el primer canal si es multicanal

total = len(data)
print(f"[INFO] Total de ventanas: {total}")

# --- Visualizar por bloques ---
i = 0
while i < total:
    bloque = data[i:i+NUM_VENTANAS].reshape(-1)
    tiempo = np.arange(len(bloque)) / sample_rate

    plt.figure(figsize=(14, 4))
    plt.plot(tiempo, bloque, color='steelblue')
    plt.title(f"Ventanas {i} a {min(i+NUM_VENTANAS-1, total-1)}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    avanzar = input("Presiona Enter para siguiente bloque, o 'q' para salir: ")
    if avanzar.lower() == 'q':
        break
    i += NUM_VENTANAS
