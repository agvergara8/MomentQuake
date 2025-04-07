# 🧠 Detección de Eventos Sísmicos con Moment (Modelo Fundacional de Series Temporales)

Este proyecto aplica el modelo fundacional **Moment**, basado en transformers, para la detección no supervisada de eventos sísmicos a partir de datos multicanal obtenidos por geófonos.

---

## 📁 Estructura del repositorio

```bash
.
├── moment_local_inference.py        # Script principal de inferencia y detección
├── pickleDataProcces.py             # Conversión de datos .pickle a .npy
├── npyView.py                       # Visualizador de datos en formato .npy
│
├── configs/
│   └── moment_base.yaml             # Configuración del modelo Moment (task: reconstruction)
│
├── momentfm/                        # Código fuente del modelo Moment (modelo, capas, datasets)
│
├── dataPickle/                      # Datos sísmicos originales (formato .pickle por geófono)
├── moment_ready_dataset.npy         # Dataset preprocesado listo para inferencia
├── detected_events_windows.npy      # Ventanas marcadas como anómalas por Moment + Isolation Forest
│
├── pca_moment_reconstruction_local.png  # Visualización PCA de reconstrucciones
│
├── requirements.txt                 # Dependencias del entorno Python
├── SeismicRequirements.txt          # Dependencias del entorno adaptadas al proyecto
├── README.md                        # Información generica de Moment
└── SEISMICREADME.md                 # Este archivo

```

---

## 📁 Estructura del repositorio
pip install -r SeismicRequirements.txt
---

## ⚙️ Instalación de requisitos

Recomendamos utilizar Python 3.9+.

Instalación de dependencias necesarias para el entorno sísmico:

```bash
pip install -r SeismicRequirements.txt
```

---

## 🚀 Flujo de trabajo principal

1. Conversión de los archivos `.pickle` originales desde `dataPickle/` a arrays `.npy` mediante `pickleDataProcces.py`.
2. Preparación del dataset final con shape `(N, 24, 512)` en `moment_ready_dataset.npy`.
3. Inferencia con Moment (`moment_local_inference.py`) usando `task: reconstruction`.
4. Detección de anomalías mediante PCA + Isolation Forest.
5. Guardado de ventanas anómalas en `detected_events_windows.npy`.
6. Visualización de las señales detectadas usando `npyView.py`.

---

## 📈 Resultados esperados

- Visualización PCA de reconstrucciones (`pca_moment_reconstruction_local.png`).
- Archivo `.npy` con ventanas detectadas como anómalas.
- Gráficas multicanal de eventos detectados.

---

## 📡 Dataset

Los datos provienen del **array de geófonos del CSIC en La Palma**, distribuidos en carpetas por geófono (1 al 8) y componente (`X`, `Y`, `Z`). Cada archivo `.pickle` contiene un objeto `obspy.Stream` con las trazas sísmicas crudas.

---

## 🧠 Modelo usado

Se utiliza el modelo fundacional **Moment**, en su modo `reconstruction`, que permite analizar ventanas multicanal y reconstruirlas sin necesidad de etiquetas.

- Las ventanas son de tamaño `(24 canales × 512 muestras)`.
- Moment analiza cada ventana como una unidad, dividiéndola en patches para procesarla con un transformer encoder.
- Las ventanas anómalas se detectan porque Moment falla más al reconstruirlas.

---

## ✏️ Autor
Álvaro García Vergara
Desarrollado en el contexto de un Trabajo de Fin de Grado (TFG) en la **Escuela Técnica Superior de Ingenieros de Telecomunicación (ETSIT-UPM)**.