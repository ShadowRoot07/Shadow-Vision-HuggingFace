import os
from core.vision import classify_image
from core.report import generate_technical_report
from utils.config import INPUT_DIR

def run_guardian_system():
    print("--- Shadow-Vision-Assistant: Iniciando Escaneo ---")
    
    # Buscamos la primera imagen disponible en data/inputs
    image_path = os.path.join(INPUT_DIR, "subaru.jpg")
    
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        return

    # Fase 1: Visión
    print(f"Analizando imagen: {image_path}...")
    tags = classify_image(image_path)
    
    if "Error" in str(tags):
        print(tags)
        return

    print(f"Objeto detectado: {tags[0]['label']} ({tags[0]['score']:.2%})")

    # Fase 2: Reporte
    print("Generando informe técnico...")
    report = generate_technical_report(tags)
    
    print("\n--- INFORME DEL GUARDIAN ---")
    print(f"OBJETO: {tags[0]['label']}")
    print(f"ANÁLISIS: {report}")
    print("----------------------------")

if __name__ == "__main__":
    run_guardian_system()

