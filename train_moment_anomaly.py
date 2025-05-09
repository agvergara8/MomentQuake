import subprocess
import os

# === Ruta al archivo YAML de configuración ===
yaml_path = "configs/anomaly_detection/end_to_end_finetuning.yaml"

# === Comando para lanzar entrenamiento ===
command = [
    "python",
    "scripts/finetuning/anomaly_detection.py",
    "--config_path", yaml_path
]

# === Ejecutar ===
print("🚀 Iniciando entrenamiento con MOMENT...")
print("Usando configuración:", yaml_path)

result = subprocess.run(command)

if result.returncode == 0:
    print("✅ Entrenamiento finalizado correctamente.")
else:
    print("❌ Error durante el entrenamiento.")
