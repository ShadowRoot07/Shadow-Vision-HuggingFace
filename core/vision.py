import requests
import os
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # 1. Limpiamos espacios en blanco accidentales
    model_id = VISION_MODEL.strip()
    
    # 2. URL del Router (Estructura recomendada para Inference API)
    # Si esta falla, probaremos la simplificada
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    print(f"DEBUG: Cargando imagen desde {image_path}")
    print(f"DEBUG: Usando VISION_MODEL: '{model_id}'")
    print(f"DEBUG: URL de destino: {api_url}")
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        
        # Agregamos un timeout para evitar que el Action se quede colgado
        response = requests.post(api_url, headers=headers, data=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        
        # Si da 404, intentamos con el endpoint del Router directamente
        if response.status_code == 404:
            router_url = f"https://router.huggingface.co/hf-inference/v1/models/{model_id}"
            response = requests.post(router_url, headers=headers, data=data, timeout=30)
            if response.status_code == 200:
                return response.json()

        return f"Error en la API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error crítico: {str(e)}"

