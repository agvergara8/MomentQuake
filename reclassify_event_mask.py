import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Parámetros ---
BASE_NAME = "CSIC_LaPalma_Geophone1_X"
DATA_FOLDER = "resultados_inferencia"

# --- Cargar errores ---
error_path = os.path.join(DATA_FOLDER, f"{BASE_NAME}_errors.npy")
print(f"[INFO] Cargando errores desde {error_path}")
all_mse = np.load(error_path)

# --- Selección de umbral ---
print("\n[INTERACTIVO] Selecciona modo de umbral:")
print("1 - Usar nuevo percentil")
print("2 - Usar valor de umbral absoluto")
modo = input("Introduce 1 o 2: ").strip()

if modo == "1":
    p = float(input("Introduce el nuevo percentil (ej. 95): "))
    threshold = np.percentile(all_mse, p)
elif modo == "2":
    threshold = float(input("Introduce el umbral absoluto (ej. 3.5): "))
else:
    print("[ERROR] Opción inválida.")
    exit(1)

print(f"[INFO] Nuevo umbral aplicado: {threshold:.4f}")
event_mask = all_mse > threshold

# --- Guardar nueva máscara ---
event_mask_path = os.path.join(DATA_FOLDER, f"{BASE_NAME}_event_mask.npy")
np.save(event_mask_path, event_mask.astype(np.uint8))
print(f"[INFO] Nueva máscara guardada en: {event_mask_path}")

# --- Reprocesar CSV enriquecido ---
csv_path = os.path.join(DATA_FOLDER, f"{BASE_NAME}_resultados.csv")
if os.path.exists(csv_path):
    print(f"[INFO] Recalculando CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if len(all_mse) > len(df):
        print("[ERROR] Más errores que filas en el CSV. Abortando.")
        exit(1)
    df_partial = df.iloc[:len(all_mse)].copy()
    df_partial["es_evento"] = event_mask.astype(int)

    df_partial.to_csv(csv_path, index=False)
    print(f"[INFO] Guardado nuevo CSV: {csv_path}")

    # --- Visualización ---
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
        plt.title(f"Eventos detectados (nuevo umbral): {BASE_NAME}")
        plt.xlabel("Tiempo")
        plt.ylabel("Error de reconstrucción")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_FOLDER, f"{BASE_NAME}_grafico.png"))
        plt.close()
        print(f"[INFO] Guardado nuevo gráfico: {BASE_NAME}_grafico.png")
    except Exception as e:
        print(f"[AVISO] Fallo en visualización: {e}")
else:
    print(f"[AVISO] No se encontró el CSV enriquecido en: {csv_path}")
