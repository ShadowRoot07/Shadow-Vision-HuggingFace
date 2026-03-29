import time
from huggingface_hub import InferenceClient
from utils.config import HF_TOKEN, VISION_MODEL

def classify_image(image_path):
    if not HF_TOKEN:
        return "Error: No se encontró el HF_TOKEN en las variables de entorno."

    model_id = VISION_MODEL.strip().replace('"', '').replace("'", "")
    
    # Inicializamos el cliente
    client = InferenceClient(model=model_id, token=HF_TOKEN)

    print(f"--- Shadow-Vision Debug ---")
    print(f"Modelo: {model_id}")
    print(f"Procesando: {image_path}")

    try:
        # IMPORTANTE: Pasamos la RUTA (image_path) en lugar de los bytes leídos.
        # Esto permite que InferenceClient detecte el mime-type (image/jpeg) por sí solo.
        response = client.image_classification(image_path)
        
        return response
            
    except Exception as e:
        error_str = str(e)
        
        # Manejo de Cold Start (Modelo cargando)
        if "loading" in error_str.lower():
            print("AVISO: El modelo se está despertando en los servidores de HF. Esperando 20s...")
            time.sleep(20)
            try:
                # Reintentamos usando la ruta
                return client.image_classification(image_path)
            except Exception as e2:
                return f"Error tras reintento de carga: {str(e2)}"
        
        return f"Error crítico: {error_str}"

