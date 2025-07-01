import pandas as pd
from tqdm import tqdm

def calcular_precision(csv_pred_path, csv_ref_path):
    # Cargar CSVs
    df_pred = pd.read_csv(csv_pred_path)
    df_ref = pd.read_csv(csv_ref_path)

    ventana_size = 512
    true_positives = 0
    false_positives = 0
    total_eventos = 0

    # Barra de progreso sobre el número de filas
    for _, row_pred in tqdm(df_pred.iterrows(), total=len(df_pred), desc="Procesando predicciones"):
        try:
            trace_name = row_pred["trace_name"]
            window_in_file = int(row_pred["window_in_file"])
            es_evento = int(row_pred["es_evento"])
        except (ValueError, TypeError):
            print(f"[IGNORADO] Fila predicción con datos inválidos: {row_pred.to_dict()}")
            continue

        if es_evento != 1:
            continue  # Solo nos interesan las predichas como evento

        total_eventos += 1

        # Buscar referencia
        ref_rows = df_ref[df_ref["trace_name"] == trace_name]
        if ref_rows.empty:
            print(f"[ADVERTENCIA] Traza no encontrada en referencia: {trace_name}")
            continue
        ref_row = ref_rows.iloc[0]

        try:
            p_arrival = int(ref_row["p_arrival_sample"])
            coda_end = int(ref_row["coda_end_sample"])
        except (ValueError, TypeError):
            print(f"[IGNORADO] Fila referencia con datos inválidos: {ref_row.to_dict()}")
            continue

        # Rango de muestras de la ventana
        start_sample = window_in_file * ventana_size
        end_sample = start_sample + ventana_size - 1

        # Comprobar solapamiento
        if end_sample >= p_arrival and start_sample <= coda_end:
            true_positives += 1
        else:
            false_positives += 1

    print("\n=== RESULTADOS ===")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    if total_eventos > 0:
        precision = true_positives / total_eventos
        print(f"Precisión: {precision:.4f}")
    else:
        print("No hay ventanas predichas como evento.")

# Ejemplo de uso:
# calcular_precision("predicciones.csv", "referencia.csv")

calcular_precision("predicciones_clean.csv", "referencia_clean.csv")
