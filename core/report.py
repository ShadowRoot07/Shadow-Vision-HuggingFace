import requests
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(tags):
    """
    Toma las etiquetas de visión y genera un texto descriptivo.
    """
    if not HF_TOKEN:
        return "Error: Token de Hugging Face no configurado."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"
    
    # Extraemos solo el nombre de la etiqueta con mayor probabilidad
    main_tag = tags[0]['label'] if isinstance(tags, list) and len(tags) > 0 else "Unknown Object"
    
    prompt = f"Technical analysis of the detected object: {main_tag}. Characteristics and state:"
    
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50, "return_full_text": False}
    }

    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text'] if isinstance(result, list) else result
    else:
        return f"Error en reporte: {response.status_code}"

