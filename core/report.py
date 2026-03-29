import requests
import json
import time
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # Limpieza del modelo
    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    # Construimos la URL manualmente para evitar intermediarios
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"System: Eres un sistema de seguridad. Resume el hallazgo: {detected_object}",
        "parameters": {"max_new_tokens": 50, "return_full_text": False}
    }

    try:
        # Petición POST directa
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        # Si el modelo está cargando (Cold Start)
        if response.status_code == 503:
            print("AVISO: Modelo en 'Cold Start'. Esperando 20s...")
            time.sleep(20)
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            # La respuesta suele ser una lista: [{'generated_text': '...'}]
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'Sin texto generado.').strip()
            return str(result)
        else:
            return f"Error API ({response.status_code}): {response.text[:50]}"

    except Exception as e:
        return f"Fallo en la Matriz (Requests): {str(e)}"

