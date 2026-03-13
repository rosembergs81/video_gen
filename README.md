# 🎬 Open-Source Video Generator — Vast.ai (Enhanced Edition)

Generador avanzado de videos de alta calidad a partir de texto o imágenes empleando modelos 100% open-source, con una interfaz web integral orientada a la producción cinematográfica y generación multi-escena.

---

## ✨ Novedades Recientes (Audit Fixes & Upgrades)
- **Generación en Cola (Queue)**: Procesamiento asíncrono robusto para soportar entornos remotos (ej: Vast.ai) y varios comandos concurrentes.
- **Monitor en Tiempo Real (GPU)**: Lectura activa de memoria VRAM (Localizada y Reservada) directo en la inferfaz web.
- **Galería Integrada**: Módulo de historia visual que carga los últimos renders disponibles.
- **Comparación A/B (Video)**: Utilidad para cargar 2 ID de generación histórica y compararlos lado-a-lado guardándolos en la base de datos local SQLite.
- **SceneGraph Activo**: Seguimiento robusto de oclusiones y posiciones espaciales durante el _prompt engineering_.
- **Performance de Base de Datos Mejorda**: Conexiones a la DB (`videogen.db`) vía Sistema Completo de Connection Pooling seguro sobre multi-hilos para la persistencia del sistema.
- **Correcciones Estructurales**: Renombrado correcto a `loras/` estándar para entornos Linux, sub-módulo centralizado `__init__.py`.

---

## 🤖 Modelos incluidos

| Modelo | Tipo | VRAM | Calidad |
|--------|------|------|---------|
| **CogVideoX-5B** | Texto → Video | ~18 GB | ⭐⭐⭐⭐⭐ |
| **CogVideoX-5B-I2V** | Imagen → Video | ~18 GB | ⭐⭐⭐⭐⭐ |
| **LTX-Video** | Texto → Video | ~10 GB | ⭐⭐⭐⭐ |
| **LTX-Video-I2V** | Imagen → Video | ~10 GB | ⭐⭐⭐⭐ |

---

## 📁 Estructura del proyecto

A diferencia de clientes simples, este generador está gobernado por una arquitectura robusta de _módulos especializados_.

```text
video_gen/
├── app.py                     # App Servidor Principal con Gradio UI (Con GPU Queue encolada)
├── generate.py                # Pipeline CLI para generación sin terminal gráfica local
├── setup.sh                   # Script de instalación para vast.ai linux/ubuntu
├── requirements.txt           # Dependencias estandarizadas
├── README.md                  # Doc
├── loras/                     # Directorio normalizado de pesos lora y settings
├── outputs/                   # Videos MP4 finalizados
└── modules/                   # Módulos CORE Internos:
    ├── __init__.py            # Exportaciones globales y APIs
    ├── action_extractor.py    # NLP Regex: Verbos y detectores de acciones textuales
    ├── database.py            # SQLite3 Thread-Pooleado Manager
    ├── gesture_templates.py   # Librería y mapping de gestos y templates cinemáticos
    ├── lora_recommender.py    # Auto-Recomendador de LoRAs de acuerdo al Contexto del Prompt
    ├── lora_scheduler.py      # Weights interpolator local
    ├── motion_interpolator.py # Catmull-Rom para control de movimiento temporal
    ├── physics_validator.py   # Validador Semántico #1: Analizador de Momentum vs FrameTime
    ├── prompt_enhancer.py     # Gestor de iluminación, cinematografica y atmósferas
    ├── scene_graph.py         # Relaciones tridimensionales relativas (Oclusiones/Contactos)
    ├── scene_parser.py        # Módulo de secuencias directas (Ej: Text1 -> Text 2)
    ├── skeletal_animator.py   # Framework de poses de humanos y bi-pedos vectorizadas 
    └── temporal_validator.py  # Gestor general de Validación y coherencia Frame-by-frame
```

---

## 🖥️ Interfaz Web (Gradio)

Al lanzar `app.py`, dispondrás de un panel de 8 pestañas modulares:
1. **🎬 Generar**: Ingesta del Prompt Simple o Estructurado (`[SCENE] -> [CHAR]`), selección de cámara, LoRAs aplicables, modelo IA, semilla (seed) parametrizada. 
2. **🎨 LoRAs**: Gestor de importación LoRA (HF hub URL), con scheduling dinámico por curvas (`fade_in`, `fade_out`, `pulse`). 
3. **🤸 Gestos**: Explorador de lenguaje corporal compatible con `gesture_templates.py`.
4. **📊 Historial**: Base de datos tabulada y recuentos de modelo via SQLite.
5. **🔍 NLP / Análisis**: Extrae verbos estructurados para pre-validar y enriquecer un script cinemático antes de gastar recursos de hardware. Score compuesto entre Momentum vs Física.
6. **🤖 LoRA Recomendador**: Lee el NLP del usuario y sugiere librerías HuggingFace conocidas de LoRAs automáticas al script escrito.
7. **🦴 Esqueleto Animador**: Interpola _poses vectoriales_. (e.g. Crouching -> Standing) transformadas a texto de red neuronal frame-por-frame.
8. **🖼️ Galería y Comparación**: *NUEVO*. Carga la galería visual a memoria o ejecuta una visualización paralela A/B para documentar estudios de arte.

---

## 🚀 Despliegue en Vast.ai — Paso a paso

### 1. Crear instancia

En [vast.ai](https://vast.ai), busca una instancia con:
- **GPU recomendada:** RTX 4090 (24 GB) · A100 40GB · H100
- **GPU mínima:** RTX 3090 (24 GB) para CogVideoX, RTX 3080 para LTX
- **Disco:** ≥ 80 GB
- **Docker Image:** `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`

### 2. Subir archivos y Ejecutar Configuración

Sube esta carpeta entera (`video_gen`) a `/workspace/video_gen`. Luego, ingresa:
```bash
cd /workspace/video_gen
bash setup.sh
```

### 3. Iniciar la App
Puedes arrancar la interface con monitoreo y cola asíncrona mediante:
```bash
# Red Glocal (Vast AI expose puerto TCP)
python app.py --host 0.0.0.0 --port 7860

# Tunnel Share (Link https:// temporal gratuito via Gradio)
python app.py --share

source /workspace/venv/bin/activate && python /workspace/video_gen/app.py --share

# Acceso Seguro:
python app.py --auth user123 claveSegura
```

*(El portal indicará "GPU: VRAM Uso..." en verde para confirmar carga correcta).*

---

## 🛠️ Uso manual desde CLI (Background Task)

Activa el ambiente virtual Python e invoca `generate.py` localmente para automatizar en lote:
```bash
# Entorno
source /workspace/venv/bin/activate
cd /workspace/video_gen

# CogVideoT2V — Alta Frecuencia/Pasos
python generate.py \
  --prompt "A warrior stands in the rain, ultra high quality slow motion." \
  --model cogvideox \
  --frames 49 \
  --fps 24 \
  --steps 50 \
  --output ./outputs/prueba_cinematica.mp4

# LTX-Video I2V — Generación desde una imágen fija inicial (JPG/PNG)
python generate.py \
  --prompt "Camera slowly pushes into the old portrait, coming alive." \
  --model ltx-i2v \
  --image retrato_viejo.jpg \
  --frames 97 \
  --steps 25 \
  --output ./outputs/retrato_vivo.mp4
```

---

## 💡 Prompts y Consejos

### Sistema Estructurado
Aunque funciona con texto directo, tú puedes forzar directivas de inyección (soportado por el `scene_parser.py`) como:
```text
[SCENE: Futuristic city at night]
[CHAR_1: Cyberpunk detective, trench coat]
  - CHAR_1.head: turning to the right sharply
[INTERACTION: Cyberpunk detective starts running]
```

### Tips de Optimizacion VRAM y Tiempo
- En RTX 3090 / 4090 bajar a 25/33 frames reduce la VRAM solicitada a ~12-14 GB reales dependiendo si usas FP16 o BF16.
- CogVideoX-5B toma entre 4 a 7 Minutos. LTX toma de 1 a 2 Minutos para la misma cantidad de tokens de entrada.
- Usa en Negative Prompt constantes como: `blurry, distorted, artifacts, text, watermark, bad quality`.
- Importa variables de autenticación para librerías cerradas (`export HF_TOKEN=hf_...` o editando `.bashrc`) previo a la descarga de AutoModels.
