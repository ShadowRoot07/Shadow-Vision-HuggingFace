import requests
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = VISION_MODEL.strip()
    
    # URL DEFINITIVA SEGÚN EL NUEVO PROTOCOLO DEL ROUTER
    # El Router actúa como pasarela hacia el v1
    api_url = f"https://router.huggingface.co/hf-inference/v1/models/{model_id}"
    
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream" # Importante para enviar binarios al Router
    }

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        print(f"DEBUG: Enviando {len(image_data)} bytes a través del Router...")
        
        response = session.post(api_url, headers=headers, data=image_data, timeout=60)

        # Si el servidor responde con HTML (como el error 410 anterior), lo detectamos
        if "doctype html" in response.text.lower():
            return f"Error de Infraestructura: El Router devolvió HTML en lugar de JSON (Status: {response.status_code})"

        if response.status_code == 200:
            return response.json()
        
        # Manejo de modelo cargando (Cold Start)
        try:
            res_json = response.json()
            if isinstance(res_json, dict) and "estimated_time" in res_json:
                wait_time = res_json["estimated_time"]
                print(f"DEBUG: Modelo despertando... esperando {wait_time}s")
                time.sleep(wait_time)
                response = session.post(api_url, headers=headers, data=image_data, timeout=60)
                return response.json()
            return f"Error API ({response.status_code}): {res_json}"
        except:
            return f"Error API ({response.status_code}): {response.text[:200]}" # Solo los primeros 200 caracteres si no es JSON
            
    except Exception as e:
        return f"Error crítico en la ejecución: {str(e)}"

