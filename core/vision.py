import requests
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # URL actualizada al nuevo Router
    api_url = f"https://api-inference.huggingface.co/models/{VISION_MODEL}"
    
    # IMPORTANTE: Añadimos "Content-Type" para que la API sepa que es una imagen
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg" 
    }

    with open(image_path, "rb") as f:
        data = f.read()
    
    # Usamos la API directamente (HF a veces redirige el router automáticamente)
    response = requests.post(api_url, headers=headers, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        # Esto nos ayudará a ver qué pasa si vuelve a fallar
        return f"Error en la API: {response.status_code} - {response.text}"

