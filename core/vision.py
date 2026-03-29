import requests
import os
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = VISION_MODEL.strip()
    
    # Probamos la URL de inferencia directa (clásica)
    # Esta es la que suele funcionar para modelos de Image Classification
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    print(f"DEBUG: Escaneando {image_path}")
    print(f"DEBUG: Intentando con API URL: {api_url}")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
    }

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Enviamos los bytes directamente
        response = requests.post(api_url, headers=headers, data=image_data, timeout=30)
        
        # Si esta URL nos devuelve 404 o 410, intentamos con la URL de respaldo del Router v1
        if response.status_code in [404, 410]:
            backup_url = f"https://api-inference.huggingface.co/pipeline/image-classification/{model_id}"
            print(f"DEBUG: Reintentando con Backup URL: {backup_url}")
            response = requests.post(backup_url, headers=headers, data=image_data, timeout=30)

        if response.status_code == 200:
            return response.json()
        
        return f"Error {response.status_code} - MSG: {response.text}"
            
    except Exception as e:
        return f"Error crítico: {str(e)}"

