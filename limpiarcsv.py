import pandas as pd

# Cargar el CSV
df = pd.read_csv("predicciones.csv")

# Reemplazar en todo el DataFrame: ejemplo, quitar '.0' de strings
df = df.astype(str).map(lambda x: x.replace(".1", ""))
df = df.astype(str).map(lambda x: x.replace(".0", ""))
df = df.astype(str).map(lambda x: x.replace(".2", ""))

# Guardar de nuevo
df.to_csv("predicciones_clean.csv", index=False)

# Cargar el CSV
df = pd.read_csv("referencia.csv")

# Reemplazar en todo el DataFrame: ejemplo, quitar '.0' de strings
df = df.astype(str).map(lambda x: x.replace(".1", ""))
df = df.astype(str).map(lambda x: x.replace(".0", ""))
df = df.astype(str).map(lambda x: x.replace(".2", ""))

# Guardar de nuevo
df.to_csv("referencia_clean.csv", index=False)

print("Reemplazo completado y archivo guardado.")
