import time
from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN."

    model_id = VISION_MODEL.strip()
    
    # Usamos el cliente oficial que gestiona la lógica de rutas interna de HF
    client = InferenceClient(model=model_id, token=HF_TOKEN)

    print(f"DEBUG: Escaneando con InferenceClient: {image_path}")
    print(f"DEBUG: Modelo objetivo: {model_id}")

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # El cliente detecta que es una tarea de clasificación por el modelo
        # Intentará contactar con el endpoint correcto automáticamente
        response = client.image_classification(image_data)
        
        return response
            
    except Exception as e:
        error_msg = str(e)
        # Si el modelo está cargando, a veces el cliente lanza una excepción con el tiempo
        if "currently loading" in error_msg.lower():
            print("DEBUG: Modelo cargando, esperando 20 segundos para reintentar...")
            time.sleep(20)
            try:
                return client.image_classification(image_data)
            except Exception as e2:
                return f"Error tras reintento: {str(e2)}"
        
        return f"Error crítico: {error_msg}"

