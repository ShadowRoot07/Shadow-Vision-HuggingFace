from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, TEXT_MODEL

def generate_technical_report(detected_object):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = TEXT_MODEL.strip().replace('"', '').replace("'", "")
    client = InferenceClient(model=model_id, token=HF_TOKEN)
    
    # Prompt optimizado para Mistral
    prompt = f"<s>[INST] Eres un dron de seguridad. Resume brevemente (1 frase) el hallazgo de: {detected_object} [/INST]"

    try:
        # El parámetro clave es stream=False para evitar el StopIteration
        output = client.text_generation(
            prompt, 
            max_new_tokens=50,
            stream=False, # <--- ESTO arregla el StopIteration
            temperature=0.7
        )
        
        return output.strip() if output else "Informe vacío."

    except Exception as e:
        # Con repr(e) seguiremos viendo si algo más se rompe
        return f"Fallo en la Matriz: {repr(e)}"

