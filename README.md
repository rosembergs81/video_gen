# 🎬 Open-Source Video Generator — Vast.ai (Enhanced Edition)

Generador avanzado de videos de alta calidad a partir de texto, imágenes o videos empleando modelos 100% open-source, con una interfaz web integral orientada a la producción cinematográfica y generación multi-escena.

---

## ✨ Novedades Recientes (Audit v3)

### 🆕 Nuevas Funcionalidades
- **🎯 Control Espacial (ControlNet-style BETA)**: Nuevo módulo `spatial_control.py` que renderiza **mapas de profundidad** y **overlays de esqueleto (OpenPose)** a partir del `SceneGraph` y `SkeletalAnimator`. Genera una imagen de control que se inyecta como primer frame del pipeline I2V, forzando coherencia posicional sin necesitar un ControlNet real.
- **⚡ Caché LRU de Segmentos**: Nuevo módulo `cache_manager.py` que computa un hash SHA-256 por segmento (prompt + semilla + parámetros). Los segmentos ya generados se recuperan del disco instantáneamente, ahorrando minutos de GPU al iterar en Story Mode.
- **🖼️ Galería Visual de LoRAs**: El gestor de LoRAs ahora soporta **thumbnails/miniaturas** y muestra un grid visual interactivo en la pestaña LoRAs, similar a CivitAI.
- **📦 Auto-descarga de Modelos**: Script `download_models.py` integrado en `setup.sh` que pre-descarga todos los modelos al ejecutar la instalación, eliminando pausas durante la primera generación.

### 🔧 Bugs Corregidos (v3)
- **Fix V2V CLI**: Corregido el modo Video-to-Video en `generate.py` donde se usaba `args.frames` en lugar de `len(video_frames)`, causando fallos en el pipeline.
- **Variable huérfana eliminada**: Removido cálculo inactivo de `jitter_mag` en `physics_validator.py`.
- **LoRA Pre-loading**: Implementado `preload_all()` en `lora_scheduler.py` para cargar todos los adaptadores LoRA antes de iniciar la generación, evitando fragmentación VRAM y pausas a mitad del rendering.

### 🛠️ Refactorizaciones (v3)
- **Reglas de validación externalizadas**: `temporal_validator.py` ahora lee las reglas de transición, inercia y momentum desde un archivo JSON editable (`temporal_rules.json`), permitiendo personalizar la "física" por estilo (realista vs anime).
- **Dataclass estático**: Reemplazada la generación dinámica `make_dataclass()` por una clase `SimpleKeyframe` definida estáticamente, mejorando rendimiento e idiomaticidad.

### Novedades anteriores (Audit v2)
- **Story Mode (Continuidad Visual)**: Generación de historias multi-escena encadenando modelos T2V y I2V con crossfade automático.
- **Soporte Video-to-Video (V2V)**: `CogVideoX-5B-V2V` para transformar videos existentes con control de fuerza (`strength`).
- **Arquitectura Modular Limpia**: Refactor de `app.py` extrayendo lógica a `pipeline_utils.py`, `lora_manager.py`, `generation.py` y `story_mode.py`.
- **Robustez y Anti-Crashes**: Protectores OOM durante generación en lote y estabilización de callbacks.
- **Connection Pooling**: Base de datos SQLite con pool de conexiones thread-safe vía `queue.Queue`.

---

## 🤖 Modelos incluidos

| Modelo | Tipo | VRAM | Calidad |
|--------|------|------|---------|
| **CogVideoX-5B** | Texto → Video | ~18 GB | ⭐⭐⭐⭐⭐ |
| **CogVideoX-5B-I2V** | Imagen → Video | ~18 GB | ⭐⭐⭐⭐⭐ |
| **CogVideoX-5B-V2V** | Video → Video | ~18 GB | ⭐⭐⭐⭐⭐ |
| **LTX-Video** | Texto → Video | ~10 GB | ⭐⭐⭐⭐ |
| **LTX-Video-I2V** | Imagen → Video | ~10 GB | ⭐⭐⭐⭐ |

---

## 📁 Estructura del proyecto

```text
video_gen/
├── app.py                     # Servidor Principal Gradio UI (10 pestañas)
├── generate.py                # Pipeline CLI para generación headless
├── download_models.py         # Pre-descarga de todos los modelos al caché
├── setup.sh                   # Script de instalación para Vast.ai (linux/ubuntu)
├── requirements.txt           # Dependencias
├── temporal_rules.json        # Reglas de validación editables (JSON)
├── README.md
├── loras/                     # Directorio de pesos LoRA y configuración
│   └── index.json             # Registro de LoRAs con thumbnails
├── outputs/                   # Videos MP4 finalizados
├── .cache/                    # Caché LRU de segmentos generados
│   └── segments/              # Frames cacheados por hash SHA-256
└── modules/                   # Módulos CORE:
    ├── __init__.py            # Exportaciones globales
    ├── generation.py          # Lógica central de generación (T2V / I2V / V2V)
    ├── story_mode.py          # Generador multi-escena con continuidad visual
    ├── pipeline_utils.py      # Caché de pipelines, catálogo MODELS, utilidades
    ├── cache_manager.py       # 🆕 Caché LRU en disco para segmentos de video
    ├── spatial_control.py     # 🆕 Render de depth maps + pose overlays (ControlNet-style)
    ├── lora_manager.py        # Gestor UI de LoRAs con thumbnails
    ├── lora_scheduler.py      # Scheduling dinámico de LoRAs por frame con curvas
    ├── lora_recommender.py    # Auto-recomendador de LoRAs por contexto de prompt
    ├── action_extractor.py    # NLP: verbos, velocidad, dirección, emoción
    ├── database.py            # SQLite thread-safe con connection pooling
    ├── gesture_templates.py   # Biblioteca de gestos cinemáticos
    ├── motion_interpolator.py # Catmull-Rom para trayectorias de cámara
    ├── physics_validator.py   # Validador de aceleración, ground, camera jitter
    ├── prompt_enhancer.py     # Iluminación, cinematografía, atmósferas
    ├── scene_graph.py         # Grafo 3D de relaciones espaciales
    ├── scene_parser.py        # Parser de secuencias de escenas
    ├── skeletal_animator.py   # Poses vectoriales + interpolación → prompts
    └── temporal_validator.py  # Validación compuesta: semántica + física + momentum
```

---

## 🖥️ Interfaz Web (Gradio)

Al lanzar `app.py`, dispondrás de **10 pestañas** modulares:

| # | Pestaña | Funcionalidad |
|:-:|---------|---------------|
| 1 | **🎬 Generar** | Prompt simple o estructurado, cámara, LoRAs, modelo IA, semilla. Caché automático de segmentos. |
| 2 | **🎨 LoRAs** | Gestor con scheduling dinámico por curvas, **galería visual con thumbnails**, JSON avanzado. |
| 3 | **🤸 Gestos** | Explorador de gestos corporales compatibles. |
| 4 | **📊 Historial** | Base de datos tabulada con estadísticas por modelo. |
| 5 | **🔍 NLP / Análisis** | Extracción de verbos, pre-validación y enriquecimiento de prompts. Score compuesto. |
| 6 | **🤖 LoRA Recomendador** | Sugerencias automáticas de LoRAs basadas en el análisis NLP del prompt. |
| 7 | **🦴 Esqueleto** | Interpolación de poses vectoriales (crouching → standing) para prompts frame-by-frame. |
| 8 | **🖼️ Galería y Comparación** | Galería visual de videos + comparación A/B lado a lado. |
| 9 | **📖 Story Mode** | Generación de historias multi-escena con continuidad visual automática. |
| 10 | **🎯 Control Espacial (BETA)** | 🆕 Renderizado de depth maps + skeleton overlays para anclar el layout espacial del video. |

---

## 🎯 Control Espacial (Nuevo)

El módulo de **Control Espacial** permite generar imágenes de control que anclan la composición espacial del video generado:

```
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
│  SceneGraph  │ →  │  Depth Map  │ →  │                  │
│ (posiciones) │    │ (brillos)   │    │  Imagen Combinada│
└─────────────┘    └─────────────┘    │  (Depth + Pose)  │ → Input para I2V
┌─────────────┐    ┌─────────────┐    │                  │
│  Skeletal    │ →  │ Pose Overlay│ →  │                  │
│  Animator    │    │ (OpenPose)  │    └──────────────────┘
└─────────────┘    └─────────────┘
```

**Flujo de uso:**
1. Configura posiciones de objetos (X, Y, Z) y selecciona una pose del esqueleto.
2. Haz clic en **Renderizar mapa de control**.
3. Usa la imagen combinada como **imagen de entrada** en la pestaña Generar con un modelo **I2V**.
4. El video resultante respetará la distribución espacial de la imagen de control.

---

## ⚡ Caché LRU de Segmentos (Nuevo)

El sistema cachea automáticamente cada segmento generado en `.cache/segments/`:

- **Hash único**: SHA-256 de `(modelo + prompt + negative + frames + guidance + steps + resolución + seed + LoRAs)`.
- **Cache Hit**: Recuperación instantánea desde disco (~10ms vs ~3-5 min de GPU).
- **LRU automático**: Si el caché supera 2 GB, los segmentos más antiguos se eliminan primero.
- **Ideal para Story Mode**: Modifica solo la última escena y las anteriores se cargan del caché.

```
Consola durante generación:
  [Cache Hit]  Usando segmento cacheado para keyframe 1
  [Cache Hit]  Usando segmento cacheado para keyframe 2
  [Cache Miss] Generado y agendado para caché keyframe 3
```

---

## 🚀 Despliegue en Vast.ai — Paso a paso

### 1. Crear instancia

En [vast.ai](https://vast.ai), busca una instancia con:
- **GPU recomendada:** RTX 4090 (24 GB) · A100 40GB · H100
- **GPU mínima:** RTX 3090 (24 GB) para CogVideoX, RTX 3080 para LTX
- **Disco:** ≥ 120 GB (mínimo 80 GB solo para un modelo)
- **Docker Image:** `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`

> ⚠️ **Espacio en disco estimado:**
> | Componente | Tamaño |
> |---|---|
> | CogVideoX-5B (T2V + V2V) | ~20 GB |
> | CogVideoX-5B-I2V | ~20 GB |
> | LTX-Video (T2V + I2V) | ~10 GB |
> | Sistema + dependencias | ~10 GB |
> | Outputs + caché LRU | ~20-40 GB |
> | **Total recomendado** | **≥ 120-150 GB** |

### 2. Subir archivos y Ejecutar Configuración

Sube la carpeta `video_gen` a `/workspace/video_gen`. Luego:
```bash
cd /workspace/video_gen
bash setup.sh
```

> ℹ️ El script `setup.sh` descarga automáticamente el modelo principal (CogVideoX-5B, ~20 GB) usando `--minimal`. Para descargar modelos adicionales:
> ```bash
> # Descargar todos los modelos (~50 GB)
> python download_models.py --all
>
> # Descargar modelos específicos
> python download_models.py --select 1 2 3
> #   [1] CogVideoX-5B (T2V + V2V)  ~20 GB
> #   [2] CogVideoX-5B-I2V           ~20 GB  
> #   [3] LTX-Video (T2V + I2V)      ~10 GB
> ```
>
> Si tienes un token de HuggingFace:
> ```bash
> export HF_TOKEN=hf_tu_token_aqui
> ```

### 3. Iniciar la App
```bash
# Red Global (Vast AI expone puerto TCP)
python app.py --host 0.0.0.0 --port 7860

# Tunnel Share (Link https:// temporal gratuito via Gradio)
python app.py --share

# Con entorno virtual
source /workspace/venv/bin/activate && python /workspace/video_gen/app.py --share

# Acceso Seguro:
python app.py --auth user123 claveSegura
```

*(El portal indicará "GPU: VRAM Uso..." para confirmar carga correcta).*

---

## 🛠️ Uso manual desde CLI

```bash
# Activar entorno
source /workspace/venv/bin/activate
cd /workspace/video_gen

# CogVideoX T2V — Alta Calidad
python generate.py \
  --prompt "A warrior stands in the rain, ultra high quality slow motion." \
  --model cogvideox \
  --frames 49 \
  --fps 24 \
  --steps 50 \
  --output ./outputs/prueba_cinematica.mp4

# LTX-Video I2V — Desde imagen fija
python generate.py \
  --prompt "Camera slowly pushes into the old portrait, coming alive." \
  --model ltx-i2v \
  --image retrato_viejo.jpg \
  --frames 97 \
  --steps 25 \
  --output ./outputs/retrato_vivo.mp4

# CogVideoX V2V — Transformar video existente
python generate.py \
  --prompt "Transform into anime style, vibrant colors" \
  --model cogvideox-v2v \
  --video input_video.mp4 \
  --v2v-strength 0.7 \
  --frames 49 \
  --output ./outputs/video_anime.mp4
```

---

## 💡 Prompts y Consejos

### Sistema Estructurado
Usa directivas de inyección (soportadas por `scene_parser.py`):
```text
[SCENE: Futuristic city at night]
[CHAR_1: Cyberpunk detective, trench coat]
  - CHAR_1.head: turning to the right sharply
[INTERACTION: Cyberpunk detective starts running]
```

### Tips de Optimización
- En RTX 3090 / 4090 bajar a 25/33 frames reduce la VRAM a ~12-14 GB reales.
- CogVideoX-5B: 4-7 min. LTX-Video: 1-2 min para la misma cantidad de tokens.
- Negative prompt recomendado: `blurry, distorted, artifacts, text, watermark, bad quality`.
- Exporta `HF_TOKEN` para modelos gated: `export HF_TOKEN=hf_...` antes de ejecutar.
- **Caché LRU**: Tras la primera generación, las iteraciones siguientes con mismos parámetros son instantáneas.
- **Control Espacial**: Usa la pestaña 🎯 para generar una imagen de referencia y forzar la composición en I2V.

### Personalización de Reglas Físicas
Edita `temporal_rules.json` para ajustar las reglas de validación temporal:
```json
{
  "transition_costs": [
    {"a": "jumping", "b": "sitting", "frames": 12},
    {"a": "running", "b": "sleeping", "frames": 24}
  ],
  "deceleration_frames": {
    "running": 6,
    "flying": 8
  }
}
```
Esto permite adaptar la validación al estilo visual (realista vs anime/caricatura).
