import pandas as pd
import matplotlib.pyplot as plt

# --- Ruta al CSV enriquecido ---
CSV_PATH = "datasets/CSIC_LaPalma_Geophone1_X_resultados.csv"

# --- Cargar el CSV ---
df = pd.read_csv(CSV_PATH)

# --- Convertir timestamps con formatos mixtos (coma o sin milisegundos) ---
df["timestamp"] = pd.to_datetime(df["timestamp"].str.replace(",", "."), format="mixed")

# --- Crear gráfico ---
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["reconstruction_error"], label="Error de reconstrucción (MSE)", color="blue")

# Eventos detectados
plt.scatter(
    df[df["es_evento"] == 1]["timestamp"],
    df[df["es_evento"] == 1]["reconstruction_error"],
    color="red", label="Evento detectado", s=20
)

plt.title("Eventos detectados sobre línea temporal")
plt.xlabel("Tiempo")
plt.ylabel("Error de reconstrucción")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
