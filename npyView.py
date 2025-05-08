import numpy as np
import matplotlib.pyplot as plt

# --- Cargar datos ---
mse = np.load("reconstruction_errors.npy")
event_mask = np.load("predicted_event_mask.npy").astype(bool)

# --- Umbral (recalculado por consistencia visual) ---
threshold = np.percentile(mse, 95)

# --- Visualización ---
plt.figure(figsize=(12, 5))
plt.plot(mse, label="Error de reconstrucción (MSE)", color='blue')
plt.axhline(threshold, color='orange', linestyle='--', label=f"Umbral ({threshold:.4f})")
plt.scatter(np.where(event_mask)[0], mse[event_mask], color='red', label="Eventos detectados", s=10)
plt.title("Errores de reconstrucción y detección de eventos")
plt.xlabel("Índice de ventana")
plt.ylabel("Error MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reconstruction_error_plot.png")
plt.show()
