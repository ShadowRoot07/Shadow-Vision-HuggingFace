import time
from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    
    prompt = f"System: Eres un dron de vigilancia. Resume el hallazgo: {detected_object}."

    try:
        # Usamos un bloque try más agresivo para ver qué llega exactamente
        output = client.text_generation(
            prompt, 
            max_new_tokens=40,
            temperature=0.5
        )
        
        # Si la salida es una lista (formato antiguo de la API), sacamos el texto
        if isinstance(output, list) and len(output) > 0:
            return output[0].get('generated_text', 'Lista vacía')
        
        # Si es un string, lo devolvemos
        return output if output else "Informe generado pero vacío."

    except Exception as e:
        # Forzamos que el error se convierta a string con repr() para ver el tipo de error
        return f"Fallo en la Matriz: {repr(e)}"

