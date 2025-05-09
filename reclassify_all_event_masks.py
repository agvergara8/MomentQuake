import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Parámetros ---
DATA_FOLDER = "resultados_inferencia"

# --- Buscar todos los errores guardados ---
errors_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_errors.npy")]

print(f"[INFO] Archivos encontrados: {len(errors_files)}")

# --- Selección de umbral ---
print("\n[INTERACTIVO] Selecciona modo de umbral para TODOS los archivos:")
print("1 - Usar nuevo percentil")
print("2 - Usar valor de umbral absoluto")
modo = input("Introduce 1 o 2: ").strip()

if modo == "1":
    p = float(input("Introduce el nuevo percentil (ej. 95): "))
elif modo == "2":
    threshold_abs = float(input("Introduce el umbral absoluto (ej. 3.5): "))
else:
    print("[ERROR] Opción inválida.")
    exit(1)

# --- Procesar cada archivo ---
for error_file in errors_files:
    base_name = error_file.replace("_errors.npy", "")
    print(f"\n[INFO] Procesando: {base_name}")

    error_path = os.path.join(DATA_FOLDER, error_file)
    all_mse = np.load(error_path)

    if modo == "1":
        threshold = np.percentile(all_mse, p)
    else:
        threshold = threshold_abs

    print(f"[INFO] Umbral aplicado: {threshold:.4f}")
    event_mask = all_mse > threshold

    # Guardar nueva máscara
    mask_path = os.path.join(DATA_FOLDER, f"{base_name}_event_mask.npy")
    np.save(mask_path, event_mask.astype(np.uint8))

    # Actualizar CSV enriquecido
    csv_path = os.path.join(DATA_FOLDER, f"{base_name}_resultados.csv")
    if not os.path.exists(csv_path):
        print(f"[AVISO] No se encontró CSV enriquecido para {base_name}. Saltando.")
        continue

    df = pd.read_csv(csv_path)
    if len(all_mse) > len(df):
        print("[ERROR] Más errores que filas en el CSV. Saltando.")
        continue

    df_partial = df.iloc[:len(all_mse)].copy()
    df_partial["es_evento"] = event_mask.astype(int)
    df_partial.to_csv(csv_path, index=False)
    print(f"[INFO] CSV actualizado: {csv_path}")

    # Visualización
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
        plt.title(f"Eventos detectados (nuevo umbral): {base_name}")
        plt.xlabel("Tiempo")
        plt.ylabel("Error de reconstrucción")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_FOLDER, f"{base_name}_grafico.png"))
        plt.close()
        print(f"[INFO] Gráfico actualizado: {base_name}_grafico.png")
    except Exception as e:
        print(f"[AVISO] Visualización fallida para {base_name}: {e}")
