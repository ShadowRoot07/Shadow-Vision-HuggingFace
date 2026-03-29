import requests
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = VISION_MODEL.strip()
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        print(f"DEBUG: Enviando {len(image_data)} bytes a {model_id}...")
        
        response = session.post(api_url, headers=headers, data=image_data, timeout=60)

        # Verificamos si la respuesta está vacía
        if not response.text:
            return f"Error: El servidor devolvió una respuesta vacía (Status: {response.status_code})"

        # Intentamos parsear solo si el status es 200
        if response.status_code == 200:
            return response.json()
        
        # Si no es 200, intentamos ver si es el error de 'loading'
        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "estimated_time" in error_data:
                wait_time = error_data["estimated_time"]
                print(f"DEBUG: Modelo despertando... esperando {wait_time}s")
                time.sleep(wait_time)
                response = session.post(api_url, headers=headers, data=image_data, timeout=60)
                return response.json()
            return f"Error API ({response.status_code}): {error_data}"
        except:
            return f"Error API ({response.status_code}): {response.text}"
            
    except Exception as e:
        return f"Error crítico en la ejecución: {str(e)}"

