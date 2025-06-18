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
    dcc.Store(id="click_info"),
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
            dcc.Graph(id="spectrogram_plot", style={"width": "48%", "display": "inline-block"})
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
    Output("click_info", "data"),
    Output("spectrogram_plot", "figure"),
    Input("error_plot", "clickData"),
    State("geophone", "value"),
    State("channel", "value")
)
def show_selected_window(clickData, geo, ch):
    if clickData is None:
        return go.Figure(), {}, go.Figure()
    try:
        point = clickData["points"][0]
        if "customdata" not in point:
            raise ValueError("customdata ausente en el punto clicado")
        
        index_global = int(click_info["index"])
        print(f"[DEBUG] Clic detectado: window_global_idx = {index_global}")
        base = f"CSIC_LaPalma_{geo}_{ch}"
        csv_path = os.path.join("resultados_inferencia", f"{base}_resultados.csv")
        df = pd.read_csv(csv_path)
        df = df[df["timestamp"].notnull()].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["window_global_idx"] = df["window_global_idx"].astype(int)
        match = df[df["window_global_idx"] == index_global]
        if match.empty:
            return go.Figure(), {}, go.Figure()
        index = match.index[0]
        print(f"[DEBUG] Índice relativo en el .npy: {index}")
    except Exception as e:
        print(f"[ERROR] Fallo al procesar clic: {e}")
        return go.Figure(), {}, go.Figure()

    base = f"CSIC_LaPalma_{geo}_{ch}"
    npy_path = os.path.join("datasets", f"{base}.npy")
    if not os.path.exists(npy_path):
        return go.Figure(), {}, go.Figure()
    data = np.load(npy_path)
    if data.ndim == 3 and data.shape[1] != 512:
        data = np.transpose(data, (0, 2, 1))
    if index >= len(data):
        return go.Figure(), {}, go.Figure()

    # Concatenar 2 ventanas antes y después
    start_idx = max(0, index - 2)
    end_idx = min(len(data), index + 3)
    segment = data[start_idx:end_idx].reshape(-1)
    ventana_offset = (index - start_idx) * 512

    # Tiempo en segundos
    sample_rate = 250  # asumir frecuencia de muestreo fija
    time_axis = np.arange(len(segment)) / sample_rate
    time_start = ventana_offset / sample_rate
    time_end = (ventana_offset + 512) / sample_rate

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time_axis, y=segment, mode="lines", name="Señal extendida"))
    fig_time.add_vline(x=time_start, line=dict(color="red", dash="dash"))
    fig_time.add_vline(x=time_end, line=dict(color="red", dash="dash"))
    fig_time.update_layout(title="Dominio del tiempo", xaxis_title="Tiempo (s)", yaxis_title="Amplitud")

    S = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=1, amin=1e-5, top_db=None)
    t_spec = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sample_rate, hop_length=HOP_LENGTH)

    fig_spec = px.imshow(
        S_db,
        x=t_spec,
        origin='lower',
        aspect='auto',
        color_continuous_scale='jet',
        labels={'x': 'Tiempo (s)', 'y': 'Frecuencia (Hz)', 'color': 'dB'},
        zmin=S_MIN, zmax=S_MAX
    )
    fig_spec.add_vline(x=time_start, line=dict(color="red", dash="dash"))
    fig_spec.add_vline(x=time_end, line=dict(color="red", dash="dash"))
    fig_spec.update_layout(
        title="Espectrograma",
        margin=dict(t=30, b=30),
        coloraxis_colorbar=dict(title="dB", x=1)
    )
    fig_spec.update_yaxes(range=[FREQ_MIN, FREQ_MAX])
    return fig_time, {"index": index_global, "rel_index": index}, fig_spec

# --- Ejecutar ---
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
