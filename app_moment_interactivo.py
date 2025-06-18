import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from io import BytesIO
import base64

# --- Configuración ---
RESULTS_DIR = "resultados_inferencia"
PREDICTION_DIR = "Moment_prediction"
WIN_LENGTH = 128
HOP_LENGTH = 64
N_FFT = 512
WINDOW = 'hann'
S_MIN = 60
S_MAX = 120
FREQ_MIN = 5
FREQ_MAX = 125

# --- Inicializar la app ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Explorador de eventos sísmicos"

# --- Geófonos y canales disponibles ---
geophones = sorted(set([f.split("_")[2] for f in os.listdir(RESULTS_DIR) if f.endswith("_resultados.csv")]))
channels = ["X", "Y", "Z"]

# --- Layout ---
app.layout = dbc.Container([
    dcc.Store(id="clickData"),
    html.H2("Explorador de eventos MOMENT"),
    html.Div([
        html.Label("Selecciona geófono:"),
        html.Br(),
        html.Label("Buscar por timestamp (ej. 2025-02-19T00:07:37,984):"),
        dcc.Input(id="timestamp_input", type="text", placeholder="Pega un timestamp", debounce=True),
        html.Button("Buscar ventana", id="buscar_btn", n_clicks=0),
        html.Br(),
        dcc.Dropdown(id="geophone", options=[{"label": g, "value": g} for g in geophones], value=geophones[0]),
        html.Label("Selecciona canal:"),
        dcc.Dropdown(id="channel", options=[{"label": c, "value": c} for c in channels], value="X"),
        html.Label("Umbral de error (valor absoluto):"),
        dcc.Input(id="umbral_input", type="number", value=50, step=0.1)
    ], style={"marginBottom": "1em"}),
    dcc.Graph(id="error_plot", style={"height": "40vh"}),
    html.Hr(),
    html.Div([
        html.H4("Ventana seleccionada"),
        html.Div([
            dcc.Graph(id="time_plot", style={"width": "48%", "display": "inline-block"}),
            dcc.Graph(id="prediction_plot", style={"width": "48%", "display": "inline-block"})  # Se añadió la predicción
        ])
    ])
], fluid=True)

# --- Callbacks ---
@app.callback(
    Output("error_plot", "figure"),
    Input("geophone", "value"),
    Input("channel", "value"),
    Input("umbral_input", "value")
)
def update_error_plot(geo, ch, threshold):
    base = f"CSIC_LaPalma_{geo}_{ch}"
    path = os.path.join(RESULTS_DIR, f"{base}_resultados.csv")
    if not os.path.exists(path):
        return go.Figure()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.replace(",", "."), errors="coerce")
        df = df[df["timestamp"].notnull()].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        x_vals = df["timestamp"]
    else:
        x_vals = list(range(len(df)))

    fig = go.Figure()
    if "timestamp" in df.columns:
        x_event = df[df["reconstruction_error"] > threshold]["timestamp"]
    else:
        x_event = df[df["reconstruction_error"] > threshold].index

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df["reconstruction_error"],
        mode="lines",
        name="Error MSE",
        line=dict(color="blue"),
        customdata=df["window_global_idx"]
    ))
    fig.add_trace(go.Scatter(
        x=x_event,
        y=df[df["reconstruction_error"] > threshold]["reconstruction_error"],
        mode="markers",
        name="Eventos",
        marker=dict(color="red", size=6),
        customdata=df[df["reconstruction_error"] > threshold]["window_global_idx"]
    ))

    fig.update_layout(title=f"Errores de reconstrucción - {base}",
                      xaxis_title="Tiempo" if "timestamp" in df.columns else "Índice de ventana",
                      yaxis_title="Error MSE",
                      annotations=[dict(text='Actualización completada', xref='paper', yref='paper', showarrow=False, x=0.99, y=0.01, font=dict(size=12, color='green'), xanchor='right', yanchor='bottom')],
                      clickmode="event+select")
    return fig

@app.callback(
    Output("time_plot", "figure"),
    Output("prediction_plot", "figure"),  # Añadido el gráfico de predicción
    Output("clickData", "data"),
    Input("error_plot", "clickData"),
    State("geophone", "value"),
    State("channel", "value")
)
def show_selected_window(clickData, geo, ch):
    if clickData is None:
        return go.Figure(), go.Figure(), {}

    try:
        base = f"CSIC_LaPalma_{geo}_{ch}"
        csv_path = os.path.join("resultados_inferencia", f"{base}_resultados.csv")
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str).str.replace(",", "."), errors="coerce")
        df = df[df["timestamp"].notnull()].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["window_global_idx"] = df["window_global_idx"].astype(int)

        timestamp_str = clickData.get("points", [{}])[0].get("x")
        ts_obj = pd.to_datetime(timestamp_str, errors="coerce")
        if pd.isnull(ts_obj):
            raise ValueError("Timestamp inválido")

        closest_row = df.iloc[(df["timestamp"] - ts_obj).abs().argsort().iloc[0]]
        index_global = int(closest_row["window_global_idx"])
        print(f"[DEBUG] Timestamp clicado: {timestamp_str}")
        print(f"[DEBUG] Ventana más cercana: window_global_idx = {index_global}")        

        base = f"CSIC_LaPalma_{geo}_{ch}"
        npy_path = os.path.join("datasets", f"{base}.npy")
        if not os.path.exists(npy_path):
            return go.Figure(), go.Figure(), {}

        data = np.load(npy_path)
        if data.ndim == 3 and data.shape[1] != 512:
            data = np.transpose(data, (0, 2, 1))
        if index_global >= len(data):
            return go.Figure(), go.Figure(), {}

        segment = data[index_global].squeeze()
        time_axis = np.arange(len(segment)) / 250  # sample rate of 250 Hz

        # Gráfico de la señal original (tiempo)
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(y=segment, mode="lines", name="Señal"))

        # Cargar y mostrar la predicción
        prediction_path = os.path.join("Moment_prediction", f"{base}_predictions.npy")
        if os.path.exists(prediction_path):
            prediction = np.load(prediction_path)
            pred_segment = prediction[index_global].squeeze()

            # Gráfico de la predicción
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(y=pred_segment, mode="lines", name="Predicción"))
        else:
            fig_pred = go.Figure()  # Si no hay predicción, crear una figura vacía

        return fig_time, fig_pred, {"window_global_idx": index_global}

    except Exception as e:
        print(f"[ERROR] Fallo al procesar clic: {e}")
        return go.Figure(), go.Figure(), {}

# --- Ejecutar ---
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False,port=8080)
