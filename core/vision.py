import requests
import os
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = VISION_MODEL.strip()
    
    # NUEVA URL DEL ROUTER (Obligatoria según el error 410)
    # Estructura: https://router.huggingface.co/hf-inference/v1/models/[MODEL_ID]
    api_url = f"https://router.huggingface.co/hf-inference/v1/models/{model_id}"
    
    print(f"DEBUG: Cargando imagen desde {image_path}")
    print(f"DEBUG: Usando VISION_MODEL: '{model_id}'")
    print(f"DEBUG: URL de destino (ROUTER): {api_url}")

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
        
        # Si sigue fallando, capturamos el JSON de error del router que es más informativo
        return f"Error en el Router: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error crítico: {str(e)}"

