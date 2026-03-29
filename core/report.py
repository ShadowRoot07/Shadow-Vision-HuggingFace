import requests
import json
import time
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    
    # URL ACTUALIZADA: El nuevo Router de Hugging Face
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Prompt optimizado para modelos tipo Instruct
    payload = {
        "inputs": f"Resumen técnico de seguridad: Se ha detectado un objeto tipo {detected_object}. Estado: Confirmado.",
        "parameters": {
            "max_new_tokens": 50,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        # Manejo de modelo cargando (Cold Start)
        if response.status_code == 503 or "loading" in response.text:
            print("AVISO: El Router indica que el modelo está cargando. Reintentando en 20s...")
            time.sleep(20)
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'Sin texto generado.').strip()
            elif isinstance(result, dict):
                return result.get('generated_text', str(result)).strip()
            return str(result)
        else:
            return f"Error en Router ({response.status_code}): {response.text[:50]}"

    except Exception as e:
        return f"Fallo en la Matriz (Router): {str(e)}"

