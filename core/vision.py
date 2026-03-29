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
    
    # Configuración de reintentos para conexiones inestables
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        print(f"DEBUG: Enviando {len(image_data)} bytes a {model_id}...")
        
        # Añadimos un pequeño delay por si el modelo está cargando
        response = session.post(
            api_url, 
            headers=headers, 
            data=image_data, 
            timeout=60  # Aumentamos el timeout a 60s
        )
        
        if response.status_code == 200:
            return response.json()
        
        # Si el modelo está cargando, Hugging Face devuelve un JSON con 'estimated_time'
        result = response.json()
        if "estimated_time" in result:
            wait_time = result["estimated_time"]
            print(f"DEBUG: Modelo cargando... esperando {wait_time}s")
            time.sleep(wait_time)
            # Reintento final después de la espera
            response = session.post(api_url, headers=headers, data=image_data, timeout=60)
            return response.json()

        return f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error de conexión: {str(e)}"

