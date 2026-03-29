import requests
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    """
    Envía una imagen a Hugging Face para identificar qué hay en ella.
    """
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN. Configura tus Secrets o el .env."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = f"https://router.huggingface.co/hf-inference/models/{VISION_MODEL}"

    with open(image_path, "rb") as f:
        data = f.read()
    
    response = requests.post(api_url, headers=headers, data=data)
    
    if response.status_code == 200:
        # Retorna una lista de etiquetas y probabilidades
        return response.json()
    else:
        return f"Error en la API: {response.status_code} - {response.text}"

