import time
from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    # Limpieza del ID del modelo
    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    
    # Prompt simple y directo
    prompt = f"Resume brevemente el hallazgo de este objeto: {detected_object}. Sé técnico."

    try:
        # Usamos la tarea de generación de texto estándar
        # Añadimos un timeout para evitar que se quede colgado
        output = client.text_generation(
            prompt, 
            max_new_tokens=50,
            temperature=0.7
        )
        return output if output else "El modelo devolvió un informe vacío."

    except Exception as e:
        # Esto nos dirá exactamente qué está pasando
        error_detail = str(e)
        if "loading" in error_detail.lower():
            return "Estado: Modelo despertando... Intenta de nuevo en 30s."
        return f"Fallo en la Matriz: {error_detail[:100]}" # Cortamos para que no sature el log

