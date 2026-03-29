import time
from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    # Usamos el cliente que ya probamos que funciona en la visión
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    
    prompt = f"System: Eres un dron de seguridad. \nUser: Resume el hallazgo: {detected_object}."

    try:
        # Forzamos stream=False y un timeout largo
        # Usamos el método genérico que sirve para casi cualquier modelo
        response = client.post(
            json={
                "inputs": prompt, 
                "parameters": {"max_new_tokens": 40, "return_full_text": False}
            }
        )
        
        # El .post() del cliente devuelve bytes, lo decodificamos
        import json
        result = json.loads(response.decode("utf-8"))

        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', 'Informe sin contenido.').strip()
        
        return str(result)

    except Exception as e:
        # Si el POST falla, intentamos el método directo como último recurso
        try:
            return client.text_generation(prompt, max_new_tokens=30)
        except:
            return f"Estado: Sensor de texto offline. Hallazgo: {detected_object}"


