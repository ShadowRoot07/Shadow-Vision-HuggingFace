import os
from dotenv import load_dotenv

# Cargar variables del archivo .env si existe
load_dotenv()

# Prioridad: 1. Variable de entorno (GitHub Secrets) | 2. Archivo .env
HF_TOKEN = os.getenv("HF_TOKEN")

# Configuración de Modelos de Hugging Face
# Usaremos Vision Transformer (ViT) para clasificar lo que ve el ojo del Guardian
VISION_MODEL = os.getenv("VISION_MODEL") or "google/vit-base-patch16-224"
HF_TOKEN = os.getenv("HF_TOKEN")
# Usaremos GPT-2 para generar el reporte técnico basado en la visión
TEXT_MODEL = os.getenv("TEXT_MODEL") or "HuggingFaceH4/zephyr-7b-beta"

# Rutas de archivos
INPUT_DIR = "data/inputs"
OUTPUT_MODEL_DIR = "models"

