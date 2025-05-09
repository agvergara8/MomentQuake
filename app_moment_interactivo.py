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
    html.H2("Explorador de eventos MOMENT"),
    html.Div([
        html.Label("Selecciona geófono:"),
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
        df = df.sort_values("timestamp")
        x_vals = df["timestamp"]
    else:
        x_vals = list(range(len(df)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=df["reconstruction_error"],
        mode="lines",
        name="Error MSE",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=df[df["reconstruction_error"] > threshold]["timestamp"] if "timestamp" in df.columns else df[df["reconstruction_error"] > threshold].index,
        y=df[df["reconstruction_error"] > threshold]["reconstruction_error"],
        mode="markers",
        name="Eventos",
        marker=dict(color="red", size=6)
    ))
    fig.update_layout(title=f"Errores de reconstrucción - {base}",
                      xaxis_title="Tiempo" if "timestamp" in df.columns else "Índice de ventana",
                      yaxis_title="Error MSE",
                      annotations=[dict(text='Actualización completada', xref='paper', yref='paper', showarrow=False, x=0.99, y=0.01, font=dict(size=12, color='green'), xanchor='right', yanchor='bottom')],
                      clickmode="event+select")
    return fig

@app.callback(
    Output("time_plot", "figure"),
    Output("spectrogram_plot", "figure"),
    Input("error_plot", "clickData"),
    State("geophone", "value"),
    State("channel", "value")
)
def show_selected_window(clickData, geo, ch):
    if clickData is None:
        return go.Figure(), go.Figure()
    try:
        index = clickData["points"][0]["pointIndex"]
    except Exception:
        return go.Figure(), go.Figure()

    base = f"CSIC_LaPalma_{geo}_{ch}"
    npy_path = os.path.join("datasets", f"{base}.npy")
    if not os.path.exists(npy_path):
        return go.Figure(), go.Figure()
    data = np.load(npy_path)
    if data.ndim == 3 and data.shape[1] != 512:
        data = np.transpose(data, (0, 2, 1))
    if index >= len(data):
        return go.Figure(), go.Figure()
    y = data[index].squeeze()

    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(y=y, mode="lines", name="Señal"))
    fig_time.update_layout(title="Dominio del tiempo", xaxis_title="Muestra", yaxis_title="Amplitud")

    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=WINDOW, center=True)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=1, amin=1e-5, top_db=None)
    fig_spec = px.imshow(
        S_db,
        origin='lower',
        aspect='auto',
        color_continuous_scale='jet',
        labels={'x': 'Tiempo (frames)', 'y': 'Frecuencia (Hz)', 'color': 'dB'},
        zmin=S_MIN, zmax=S_MAX
    )
    fig_spec.update_layout(
        title="Espectrograma",
        margin=dict(t=30, b=30),
        coloraxis_colorbar=dict(title="dB", x=1)
    )
    fig_spec.update_yaxes(range=[FREQ_MIN, FREQ_MAX])
    fig_spec.update_yaxes(range=[FREQ_MIN, FREQ_MAX])
    return fig_time, fig_spec

# --- Ejecutar ---
if __name__ == "__main__":
    app.run(debug=True)
