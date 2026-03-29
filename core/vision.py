import requests
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # CAMBIO CRÍTICO: Nueva estructura de URL usando el Router
    # De: https://api-inference.huggingface.co/models/...
    # A:  https://router.huggingface.co/hf-inference/v1/models/...
    
    api_url = f"https://router.huggingface.co/hf-inference/v1/models/{VISION_MODEL}"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        
        response = requests.post(api_url, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error en la API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error inesperado: {str(e)}"

