import requests
import os
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # --- CAMBIO TEMPORAL PARA TEST ---
    # Forzamos el ID exacto del modelo de Google para descartar errores en el Secret
    model_id = "google/vit-base-patch16-224" 
    # ---------------------------------
    
    api_url = f"https://router.huggingface.co/hf-inference/v1/models/{model_id}"
    
    print(f"DEBUG: Cargando imagen desde {image_path}")
    print(f"DEBUG: PROBANDO MODELO FORZADO: '{model_id}'")
    print(f"DEBUG: URL ROUTER: {api_url}")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        
        response = requests.post(api_url, headers=headers, data=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        
        # Si da 404 aquí, el problema es que el Router requiere un formato distinto para este modelo
        return f"Error en el Router: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error crítico: {str(e)}"

