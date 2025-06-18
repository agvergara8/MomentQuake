import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from obspy.core.stream import Stream

# --- Configuración ---
PICKLE_PATH = "dataPickle/CSIC_LaPalma_Geophone1_X/13-Feb-2025 at 00.00.32.pickle"
CANAL = 0  # 0: X, 1: Y, 2: Z
FS = 250  # Hz
START_TIME = datetime.strptime("13-Feb-2025 00:00:32", "%d-%b-%Y %H:%M:%S")

# --- Cargar .pickle ---
with open(PICKLE_PATH, "rb") as f:
    stream = pickle.load(f)
    if not isinstance(stream, Stream):
        print(f"[ERROR] El archivo no es un Stream válido.")
        exit()
    data = np.stack([tr.data for tr in stream], axis=1)

# --- Seleccionar canal ---
if CANAL >= data.shape[1]:
    print(f"[ERROR] El canal {CANAL} no existe en el archivo.")
    exit()

señal = data[:, CANAL]
tiempo = [START_TIME + timedelta(seconds=i / FS) for i in range(len(señal))]

# --- Visualizar ---
plt.figure(figsize=(14, 4))
plt.plot(tiempo, señal, color='steelblue')
plt.title(f"Señal completa del canal {CANAL} en {os.path.basename(PICKLE_PATH)}")
plt.xlabel("Hora")
plt.ylabel("Amplitud")
plt.grid(True)

# Formato del eje X
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()
