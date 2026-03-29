import json
from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    
    # Prompt directo
    prompt = f"Resume brevemente este hallazgo de seguridad: {detected_object}"

    try:
        # Usamos post() directamente al endpoint de tareas para evitar el iterador
        # Esto nos devuelve los bytes de la respuesta JSON
        response_bytes = client.post(
            json={"inputs": prompt, "parameters": {"max_new_tokens": 50}},
            model=model_id
        )
        
        # Decodificamos el JSON manualmente
        result = json.loads(response_bytes.decode("utf-8"))
        
        # La API de texto suele devolver una lista: [{'generated_text': '...'}]
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', 'Informe sin texto.')
        
        return str(result)

    except Exception as e:
        return f"Fallo en la Matriz (Direct Post): {repr(e)}"

