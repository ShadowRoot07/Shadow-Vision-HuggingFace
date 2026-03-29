from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object): # <--- Asegúrate que se llame así
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # Usamos un modelo robusto para el reporte
    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    
    prompt = f"System: Eres un asistente de seguridad Cyberpunk. \nUser: Resume el hallazgo: {detected_object}."

    try:
        # Importante: para modelos de chat/texto usamos text_generation
        response = client.text_generation(prompt, max_new_tokens=100)
        return response
    except Exception as e:
        return f"Error en reporte: {str(e)}"

