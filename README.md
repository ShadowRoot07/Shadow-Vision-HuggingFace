# 👁️ Shadow-Vision-HuggingFace
> **Status:** Active Surveillance | **Alias:** ShadowRoot07

Un sistema de visión artificial ligero diseñado para ejecutarse en entornos restringidos, utilizando la infraestructura de **Hugging Face Inference API** y **GitHub Actions** como orquestador de seguridad.



## 🛠️ Tech Stack
- **Engine:** Python 3.10+
- **Vision:** ViT (Vision Transformer) vía `google/vit-base-patch16-224`
- **Logic:** `InferenceClient` & `Requests` (Hybrid Router-Ready)
- **CI/CD:** GitHub Actions (Surveillance Automation)
- **Editor:** NeoVim (Nvim)
- **Environment:** Termux (Android)

## 📡 Cómo funciona
El "Guardián" escanea imágenes depositadas en `data/inputs/`. Utiliza un modelo de clasificación para identificar intrusos y un modelo de lenguaje (Gemma/Mistral) para generar un informe técnico del hallazgo.

1. **Captura:** La imagen se carga al repositorio.
2. **Análisis:** El Workflow se dispara, enviando la data al Router de Hugging Face.
3. **Reporte:** El bot genera un informe y realiza un *auto-commit* con los resultados.

## 📁 Estructura del Proyecto
```bash
.
├── core/
│   ├── vision.py      # Lógica de detección de objetos
│   └── report.py      # Generador de informes técnicos
├── data/
│   ├── inputs/        # Imágenes a procesar
│   └── reports/       # Bitácora de hallazgos (Auto-generated)
├── utils/
│   └── config.py      # Orquestación de Model IDs y Tokens
└── main.py            # Punto de entrada del Guardián
```

## 💻 Entorno de Desarrollo:
Este proyecto fue desarrollado íntegramente desde una terminal Termux en un dispositivo móvil ZTE, utilizando NeoVim como entorno de edición principal. Programación móvil pura.
