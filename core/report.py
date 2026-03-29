from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    client = InferenceClient(model=TEXT_MODEL, token=HF_TOKEN)
    
    prompt = f"Resume de forma técnica y breve el hallazgo de un objeto tipo: {detected_object}. Estado: Detectado."

    try:
        # Usamos text_generation que es el método estándar
        response = client.text_generation(prompt, max_new_tokens=50)
        return response
    except Exception as e:
        return f"Error en reporte: {str(e)}"

