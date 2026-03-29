import requests
import os
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # Limpiamos el ID
    model_id = VISION_MODEL.strip()
    
    # URL Simplificada del Router (Sin el prefijo v1/hf-inference)
    api_url = f"https://router.huggingface.co/models/{model_id}"
    
    print(f"DEBUG: Escaneando {image_path}")
    print(f"DEBUG: Intentando con ROUTER URL: {api_url}")

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
        
        # Esto es CLAVE: Si falla, queremos ver el cuerpo del error detallado
        try:
            error_detail = response.json()
        except:
            error_detail = response.text

        return f"Error {response.status_code} - Detalle: {error_detail}"
            
    except Exception as e:
        return f"Error crítico: {str(e)}"

