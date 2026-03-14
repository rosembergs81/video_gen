"""
🎬 Open-Source Video Generator — Vast.ai Ready
════════════════════════════════════════════════
Models   : CogVideoX-5B · LTX-Video
Features :
  1.  Scene parser        — keyframes secuenciales (texto natural o estructurado)
  2.  Character system    — anotación de partes del cuerpo por sujeto
  3.  Motion interpolator — transiciones suaves (Catmull-Rom splines)
  4.  Scene graph         — tracking multi-sujeto y oclusión
  5.  Camera path         — trayectorias de cámara con keyframes
  6.  Gesture templates   — biblioteca de gestos manos/cuerpo/cara
  7.  Dynamic LoRA sched. — pesos de LoRA por frame con curvas de intensidad
  8.  Prompt enhancer     — inyección de contexto cinematográfico
  9.  Temporal validator  — coherencia semántica + física + momentum
 10.  Database backend    — historial SQLite + presets + comparaciones
 11.  Action extractor    — NLP: verbo + modificadores + body-parts
 12.  LoRA recommender    — sugerencias automáticas basadas en acciones
 13.  Skeletal animator   — esqueletos + poses + interpolación → prompts
 14.  Physics validator   — aceleración, ground, jitter de cámara
"""

import argparse, json, time, uuid, os
from pathlib import Path

import torch
import gradio as gr
import numpy as np
from PIL import Image

# ── Internal modules ──────────────────────────────────────────────────────────
from modules.scene_parser        import SceneParser, BODY_PARTS
from modules.scene_graph          import SceneGraph
from modules.motion_interpolator  import MotionInterpolator, CameraPath, CameraKeyframe
from modules.gesture_templates    import (
    GESTURE_TEMPLATES, GESTURE_CATEGORIES,
    gesture_to_prompt, list_gestures,
)
from modules.lora_scheduler       import DynamicLoRASchedule, CURVES
from modules.prompt_enhancer      import (
    PromptEnhancer,
    LIGHTING_OPTIONS, CINEMATOGRAPHY_OPTIONS,
    QUALITY_OPTIONS, ATMOSPHERE_OPTIONS, GENRE_OPTIONS,
    NEGATIVE_PRESETS_LIST,
)
from modules.temporal_validator   import TemporalCoherenceValidator
from modules.database             import VideoGenDB, GenerationRecord, Preset
# ── New modules ───────────────────────────────────────────────────────────────
from modules.action_extractor     import ActionExtractor
from modules.lora_recommender     import LoRARecommender
from modules.skeletal_animator    import (
    SkeletalAnimator, SKELETON_TEMPLATES, POSE_LIBRARY,
)
from modules.physics_validator    import PhysicsValidator
from modules.lora_manager     import add_lora, remove_loras, refresh_loras, load_lora_index, get_lora_choices
from modules.generation       import generate_video as _generate_video_core
from modules.story_mode       import generate_story as _generate_story_core
from modules.pipeline_utils       import MODELS, HF_TOKEN

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("outputs");  OUTPUT_DIR.mkdir(exist_ok=True)
LORAS_DIR   = Path("loras");    LORAS_DIR.mkdir(exist_ok=True)
LORAS_INDEX = LORAS_DIR / "index.json"

CAMERA_PRESETS = [
    "—", "zoom_in", "zoom_out", "pan_left_to_right", "orbit",
    "drone_descend", "steadicam_forward",
]

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────
_parser    = SceneParser()
_enhancer  = PromptEnhancer()
_validator   = TemporalCoherenceValidator()
_db          = VideoGenDB()
_extractor   = ActionExtractor()
_recommender = LoRARecommender(db=_db)
_physics     = PhysicsValidator()

# ── Wrappers (Inyección de dependencias) ──────────────────────────────────────

def generate_video(*args, **kwargs):
    kwargs["deps"] = {
        "db": _db, "parser": _parser, "validator": _validator,
        "enhancer": _enhancer, "load_lora_index": load_lora_index,
        "MODELS": MODELS
    }
    return _generate_video_core(*args, **kwargs)

def generate_story(*args, **kwargs):
    kwargs["deps"] = {
        "db": _db, "enhancer": _enhancer,
    }
    return _generate_story_core(*args, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Preset management
# ─────────────────────────────────────────────────────────────────────────────

def save_preset_fn(name, category, model_key, prompt, negative_preset,
                   lighting, cinematography, quality, atmosphere, genre, notes):
    if not name.strip():
        return "⚠️ El nombre del preset no puede estar vacío."
    _db.save_preset(Preset(
        name=name.strip(), category=category,
        model=model_key, prompt=prompt, negative=negative_preset,
        motion_tags=dict(lighting=lighting, cinematography=cinematography,
                         quality=quality, atmosphere=atmosphere, genre=genre),
        notes=notes,
    ))
    return f"✅ Preset '{name}' guardado."

def load_preset_fn(preset_name: str):
    p = _db.get_preset(preset_name)
    if not p:
        return [gr.update()] * 8
    mt = p.motion_tags
    return (
        p.prompt,
        p.negative or "standard",
        mt.get("lighting",       "—"),
        mt.get("cinematography", "—"),
        mt.get("quality",        "—"),
        mt.get("atmosphere",     "—"),
        mt.get("genre",          "—"),
        p.notes,
    )

def preset_choices():
    return [p.name for p in _db.list_presets()]

# ─────────────────────────────────────────────────────────────────────────────
# Temporal validator UI
# ─────────────────────────────────────────────────────────────────────────────

def validate_prompt_fn(prompt, num_frames):
    """Full analysis: scene parse + action extract + semantic + physics + momentum."""
    return _validator.full_analysis(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Action Extractor UI callbacks
# ─────────────────────────────────────────────────────────────────────────────

def extract_actions_fn(prompt):
    """Extract and display structured actions from the prompt."""
    actions = _extractor.extract(prompt)
    if not actions:
        return "No se detectaron acciones reconocibles en el prompt."
    return _extractor.summarize(actions)


def enhance_from_actions_fn(prompt):
    """Use extracted actions to build an enhanced prompt."""
    actions = _extractor.extract(prompt)
    if not actions:
        return prompt
    action_str = _extractor.to_prompt_enhancement(actions)
    return f"{prompt}, {action_str}" if action_str else prompt


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Recommender UI callbacks
# ─────────────────────────────────────────────────────────────────────────────

def recommend_loras_fn(prompt, model_key, num_frames):
    """Auto-recommend LoRAs based on prompt actions + style."""
    user_loras = load_lora_index()
    recs = _recommender.recommend_from_prompt(
        prompt=prompt,
        model_key=model_key,
        total_frames=int(num_frames),
        top_n=6,
        user_loras=user_loras,
    )
    return _recommender.to_markdown(recs)


def apply_recommended_loras_fn(prompt, model_key, num_frames):
    """Generate LoRA schedule JSON from recommendations, ready to paste."""
    user_loras = load_lora_index()
    recs = _recommender.recommend_from_prompt(
        prompt=prompt,
        model_key=model_key,
        total_frames=int(num_frames),
        top_n=4,
        user_loras=user_loras,
    )
    schedule = _recommender.to_schedule_json(recs, total_frames=int(num_frames))
    return (
        _recommender.to_markdown(recs),
        json.dumps(schedule, indent=2, ensure_ascii=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skeletal Animator UI callbacks
# ─────────────────────────────────────────────────────────────────────────────

def skeleton_preview_fn(skeleton_type, pose_a, pose_b, t_val):
    """Preview interpolated pose between two named poses."""
    try:
        anim = SkeletalAnimator(skeleton_type)
        desc = anim.interpolate_to_prompt(pose_a, pose_b, float(t_val))
        return f"**Pose interpolada (t={t_val:.2f}):**  \n`{desc}`"
    except Exception as e:
        return f"⚠️ Error: {e}"


def skeleton_sequence_fn(skeleton_type, poses_str, num_frames):
    """Build animation sequence from comma-separated pose names."""
    anim    = SkeletalAnimator(skeleton_type)
    poses   = [p.strip() for p in poses_str.split(",") if p.strip()]
    valid   = [p for p in poses if p in anim.list_poses()]
    invalid = [p for p in poses if p not in anim.list_poses()]

    if not valid:
        all_poses = ", ".join(anim.list_poses())
        return f"⚠️ Ninguna pose válida. Disponibles: `{all_poses}`"

    md = anim.sequence_markdown(valid, total_frames=int(num_frames))
    if invalid:
        md += f"\n\n⚠️ Poses no reconocidas (ignoradas): `{', '.join(invalid)}`"
    return md


def skeleton_to_prompt_fn(skeleton_type, poses_str, t_val):
    """Get the prompt string for a specific time in the animation."""
    anim  = SkeletalAnimator(skeleton_type)
    poses = [p.strip() for p in poses_str.split(",") if p.strip() in anim.list_poses()]
    if not poses:
        return "⚠️ No hay poses válidas."
    flat  = anim.interpolate_sequence(poses, float(t_val))
    desc  = anim.pose_to_prompt(flat)
    return desc


def list_skeleton_poses_fn(skeleton_type):
    """Return available poses for a skeleton type."""
    try:
        anim = SkeletalAnimator(skeleton_type)
        poses = anim.list_poses()
        return f"**Poses disponibles para `{skeleton_type}`:**\n" + ", ".join(f"`{p}`" for p in poses)
    except Exception as e:
        return f"⚠️ {e}"

# ─────────────────────────────────────────────────────────────────────────────
# History UI
# ─────────────────────────────────────────────────────────────────────────────

def get_history_table():
    return _db.history_as_table(limit=100)

def delete_gen(gen_id_str):
    try:
        _db.delete_generation(int(gen_id_str))
        return "🗑️ Generación eliminada.", get_history_table()
    except Exception as e:
        return f"⚠️ Error: {e}", get_history_table()

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.gen-btn { font-size: 1.1em !important; }
#prompt-preview textarea {
    font-family: monospace; font-size: 0.83em; color: #a78bfa;
}
.tab-nav button { font-size: 0.9em; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# UI BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    lora_choices_init = get_lora_choices(load_lora_index())
    preset_names_init = preset_choices()
    gesture_list      = ["—"] + list(GESTURE_TEMPLATES.keys())

    with gr.Blocks(
        title="Video Gen — Vast.ai",
        theme=gr.themes.Soft(primary_hue="violet"),
        css=CSS,
    ) as demo:

        def gpu_monitor():
            if not torch.cuda.is_available():
                return "**GPU:** No disponible"
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                return f"**VRAM Uso:** Reservada: `{reserved:.2f} GB` | Localizada: `{allocated:.2f} GB`"
            except Exception:
                return "**GPU:** Error leyendo memoria"

        with gr.Row():
            gr.Markdown(
                "# 🎬 Open-Source Video Generator\n"
                "**CogVideoX · LTX-Video · Scene Parser · LoRA Scheduler · "
                "Gestures · Prompt Enhancer · DB History**"
            )
            gpu_status_md = gr.Markdown(value="**GPU:** Inicializando...", elem_classes="gpu-monitor")
            
            # Compatible with Gradio 5+
            timer = gr.Timer(2)
            timer.tick(gpu_monitor, outputs=gpu_status_md)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — GENERATE
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🎬 Generar"):
                with gr.Row(equal_height=False):

                    # ── LEFT ──────────────────────────────────────────────
                    with gr.Column(scale=1, min_width=420):

                        model_key  = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value=list(MODELS.keys())[0],
                            label="🤖 Modelo",
                        )
                        mode_badge = gr.Markdown("**Modo:** Texto → Video")
                        input_image = gr.Image(
                            label="🖼️ Imagen de entrada (solo modelos I2V)",
                            type="numpy", visible=False,
                        )
                        input_video = gr.Video(
                            label="🎞️ Video de entrada (solo modelo V2V)",
                            visible=False,
                        )
                        v2v_strength = gr.Slider(
                            0.1, 1.0, 0.7, step=0.05,
                            label="🎚️ Fuerza de transformación V2V (0.1=sutil, 1.0=completa)",
                            visible=False,
                        )

                        # ── Scene Prompt ───────────────────────────────
                        with gr.Accordion("📖 Prompt / Escenas", open=True):
                            gr.Markdown(
                                "Escribe un prompt simple **o** una secuencia con `→` / `then` / `luego`  \n"
                                "También puedes usar sintaxis estructurada con `[SCENE:]`, `[CHAR_1:]`, `[INTERACTION:]`"
                            )
                            raw_prompt = gr.Textbox(
                                label="Prompt / Escenas",
                                placeholder=(
                                    "Ejemplo simple:\n"
                                    "A lion starts roaring → a bird flies toward him → "
                                    "the bird lands gently on the lion's head → "
                                    "the lion slowly turns to look at the bird\n\n"
                                    "Ejemplo estructurado:\n"
                                    "[SCENE: Golden savanna at sunset]\n"
                                    "[CHAR_1: Lion, roaring]\n"
                                    "  - CHAR_1.head: turning left, mouth open\n"
                                    "[CHAR_2: Bird, graceful]\n"
                                    "  - CHAR_2.wings: flapping\n"
                                    "[INTERACTION: Bird lands on Lion's head]"
                                ),
                                lines=8,
                            )
                            validate_btn = gr.Button("🔍 Validar secuencia", size="sm")
                            validation_md = gr.Markdown()

                        # ── Prompt Enhancement ────────────────────────
                        with gr.Accordion("✨ Mejora Cinematográfica", open=False):
                            with gr.Row():
                                lighting       = gr.Dropdown(LIGHTING_OPTIONS,
                                    value="—", label="💡 Iluminación")
                                cinematography = gr.Dropdown(CINEMATOGRAPHY_OPTIONS,
                                    value="—", label="🎥 Cinematografía")
                            with gr.Row():
                                quality        = gr.Dropdown(QUALITY_OPTIONS,
                                    value="—", label="⭐ Calidad")
                                atmosphere     = gr.Dropdown(ATMOSPHERE_OPTIONS,
                                    value="—", label="🌫️ Atmósfera")
                            with gr.Row():
                                genre          = gr.Dropdown(GENRE_OPTIONS,
                                    value="—", label="🎭 Género")
                                negative_preset= gr.Dropdown(NEGATIVE_PRESETS_LIST,
                                    value="standard", label="🚫 Negative preset")

                        # ── Camera Path ───────────────────────────────
                        with gr.Accordion("📷 Cámara", open=False):
                            camera_preset = gr.Dropdown(
                                choices=CAMERA_PRESETS,
                                value="—",
                                label="Preset de trayectoria (Catmull-Rom spline)",
                            )
                            gr.Markdown(
                                "Para control avanzado, usa el editor de keyframes "
                                "de cámara en la pestaña **Avanzado**."
                            )

                        # ── Gestures ──────────────────────────────────
                        with gr.Accordion("🤸 Gestos", open=False):
                            with gr.Row():
                                gesture_name = gr.Dropdown(
                                    choices=gesture_list,
                                    value="—", label="Gesto a aplicar",
                                )
                                gesture_t = gr.Slider(
                                    0.0, 1.0, 0.5, step=0.05,
                                    label="Fase del gesto (0=inicio, 1=fin)",
                                )
                            gesture_preview = gr.Markdown()

                        # ── LoRAs (static) ────────────────────────────
                        active_loras = gr.CheckboxGroup(
                            choices=lora_choices_init, value=[],
                            label="🎨 LoRAs activas  →  gestiónalas en pestaña LoRAs",
                        )

                        # ── Generation params ─────────────────────────
                        with gr.Accordion("⚙️ Parámetros", open=False):
                            with gr.Row():
                                num_frames = gr.Slider(8, 97, 49, step=8, label="Frames")
                                fps        = gr.Slider(8, 30, 24, step=1,  label="FPS")
                            with gr.Row():
                                width  = gr.Slider(256, 1280, 720, step=64, label="Ancho px")
                                height = gr.Slider(256,  720, 480, step=64, label="Alto px")
                            with gr.Row():
                                guidance = gr.Slider(1.0, 15.0, 6.0, step=0.5,
                                                     label="Guidance scale")
                                steps    = gr.Slider(10, 50, 50, step=5,
                                                     label="Pasos")
                            seed = gr.Number(-1, label="Seed (-1 = aleatorio)", precision=0)

                        # ── Crossfade (interpolación entre keyframes) ─────
                        with gr.Accordion("🔀 Interpolación entre Keyframes", open=False):
                            crossfade_frames = gr.Slider(
                                0, 16, 4, step=1,
                                label="Frames de crossfade (0 = sin transición)",
                            )
                            gr.Markdown(
                                "Cuando tu prompt tiene múltiples escenas (`→` o `then`), la IA genera cada segmento por separado. "
                                "El **crossfade** mezcla suavemente los últimos N frames de un segmento con los primeros N del siguiente, "
                                "creando transiciones cinematográficas fluidas en lugar de cortes abruptos.\n\n"
                                "- **0**: Sin interpolación (corte directo)\n"
                                "- **2-4**: Transición rápida y sutil\n"
                                "- **8-16**: Transición larga y suave (estilo fade)"
                            )

                        # ── Presets ───────────────────────────────────
                        with gr.Accordion("💾 Presets", open=False):
                            with gr.Row():
                                preset_sel = gr.Dropdown(
                                    choices=preset_names_init,
                                    label="Cargar preset", interactive=True,
                                )
                                load_preset_btn = gr.Button("⬇️ Cargar", size="sm")
                            with gr.Row():
                                preset_name_in = gr.Textbox(
                                    placeholder="Nombre del preset…", label="Guardar como")
                                preset_cat_in  = gr.Dropdown(
                                    choices=["general","acción","naturaleza",
                                             "sci-fi","fantasy","romance"],
                                    value="general", label="Categoría",
                                )
                            preset_notes_in = gr.Textbox(
                                placeholder="Notas opcionales…", label="Notas", lines=2,
                            )
                            save_preset_btn = gr.Button("💾 Guardar preset", size="sm")
                            preset_status   = gr.Markdown()

                        gen_btn = gr.Button(
                            "🎬 Generar video", variant="primary",
                            elem_classes="gen-btn",
                        )

                    # ── RIGHT ─────────────────────────────────────────
                    with gr.Column(scale=1, min_width=400):
                        output_video = gr.Video(label="🎥 Video generado")
                        output_info  = gr.Markdown()
                        val_out_md   = gr.Markdown(label="Validación")

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — LoRAs
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🎨 LoRAs"):
                gr.Markdown("""
### Gestor de LoRAs con Scheduling Dinámico
Añade LoRAs y configura en qué **rango de frames** aplican y con qué **curva de intensidad**.
Esto permite, por ejemplo, que una LoRA de zoom-in solo afecte los primeros 20 frames.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### ➕ Añadir LoRA")
                        l_name   = gr.Textbox(label="Nombre",
                                               placeholder="Zoom In Motion")
                        l_source = gr.Textbox(
                            label="Fuente (HF repo o path local)",
                            placeholder="guoyww/animatediff-motion-lora-zoom-in")
                        l_cat    = gr.Dropdown(
                            choices=["estilo","movimiento","personaje",
                                     "iluminación","cámara","otro"],
                            value="estilo", label="Categoría")
                        l_scale  = gr.Slider(0.1, 1.5, 0.8, step=0.05,
                                              label="Escala base")
                        l_wname  = gr.Textbox(label="weight_name (opcional)")

                        gr.Markdown("**🕐 Rango de frames activos (scheduling dinámico)**")
                        with gr.Row():
                            l_fstart = gr.Number(0,  label="Frame inicio", precision=0)
                            l_fend   = gr.Number(-1, label="Frame fin (-1=hasta el final)",
                                                  precision=0)
                        l_curve  = gr.Dropdown(
                            choices=list(CURVES.keys()),
                            value="constant",
                            label="Curva de intensidad",
                        )
                        gr.Markdown(
                            "- `constant` — peso fijo durante todo el rango\n"
                            "- `fade_in` — empieza en 0, termina en escala máxima\n"
                            "- `fade_out` — empieza en escala máxima, termina en 0\n"
                            "- `pulse` — pico en el centro del rango\n"
                            "- `fade_in_out` — sube y baja suavemente"
                        )
                        add_lora_btn = gr.Button("➕ Añadir", variant="primary")
                        lora_add_status = gr.Markdown()

                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 LoRAs registradas")
                        lora_list  = gr.CheckboxGroup(
                            choices=lora_choices_init, value=[],
                            label="Selecciona para eliminar",
                        )
                        with gr.Row():
                            remove_lora_btn = gr.Button("🗑️ Eliminar", variant="stop")
                            refresh_lora_btn = gr.Button("🔄 Refrescar")
                        lora_rm_status = gr.Markdown()

                        gr.Markdown("#### ⚡ Schedule personalizado (JSON avanzado)")
                        lora_schedule_json = gr.Textbox(
                            label="JSON de schedule (opcional)",
                            placeholder=(
                                '[\n'
                                '  {"name":"zoom_in","source":"guoyww/animatediff-motion-lora-zoom-in",'
                                '"scale":0.9,"frame_start":0,"frame_end":20,"curve":"fade_out"},\n'
                                '  {"name":"anime","source":"/loras/anime.safetensors",'
                                '"scale":0.8,"frame_start":25,"frame_end":49,"curve":"fade_in"}\n'
                                ']'
                            ),
                            lines=10,
                        )

                gr.Markdown("""
---
#### 📖 LoRAs de video recomendadas

| Nombre sugerido | Fuente HuggingFace | Categoría |
|---|---|---|
| Zoom In | `guoyww/animatediff-motion-lora-zoom-in` | cámara |
| Zoom Out | `guoyww/animatediff-motion-lora-zoom-out` | cámara |
| Pan Left | `guoyww/animatediff-motion-lora-pan-left` | cámara |
| Pan Right | `guoyww/animatediff-motion-lora-pan-right` | cámara |
| Tilt Up | `guoyww/animatediff-motion-lora-tilt-up` | cámara |
| Rolling CW | `guoyww/animatediff-motion-lora-rolling-clockwise` | movimiento |
| CogVideoX Movie | `TheDenouement/cogvideox-movie-lora` | estilo |
                """)

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — GESTURES REFERENCE
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🤸 Gestos"):
                gr.Markdown("### Biblioteca de Gestos")
                for cat_label, names in GESTURE_CATEGORIES.items():
                    with gr.Accordion(cat_label, open=False):
                        for gname in names:
                            g = GESTURE_TEMPLATES[gname]
                            with gr.Row():
                                gr.Markdown(
                                    f"**{gname}** — *{g.category}*  \n"
                                    f"Loop: {'✅' if g.loop else '❌'}  |  "
                                    f"Duración relativa: {g.duration_hint}x  \n"
                                    f"Descripción completa:  \n`{g.full_description()}`"
                                )

            # ══════════════════════════════════════════════════════════════
            # TAB 4 — HISTORY & STATS
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📊 Historial"):
                gr.Markdown("### Historial de generaciones")
                with gr.Row():
                    refresh_hist_btn = gr.Button("🔄 Refrescar")
                    hist_del_id  = gr.Number(label="ID a eliminar", precision=0)
                    hist_del_btn = gr.Button("🗑️ Eliminar", variant="stop")
                    hist_status  = gr.Markdown()

                history_table = gr.DataFrame(
                    value=get_history_table(),
                    headers=VideoGenDB.HISTORY_COLUMNS,
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    stats_btn = gr.Button("📈 Ver estadísticas")
                    stats_md  = gr.Markdown()

            # ══════════════════════════════════════════════════════════════
            # TAB 5 — NLP ANALYSIS (Action Extractor + Temporal Validator)
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🔍 NLP / Análisis"):
                gr.Markdown("""
### 🔍 Extractor de Acciones  +  Validador Temporal v2
Extrae **verbos estructurados**, modificadores de velocidad/dirección/emoción y partes del cuerpo
desde tu prompt. El validador ahora incluye **física**, **momentum** e integración con ActionExtractor.
                """)

                nlp_prompt_in = gr.Textbox(
                    label="Prompt a analizar",
                    placeholder=(
                        "A lion slowly turns its head to the left, roaring aggressively → "
                        "a bird flies quickly toward him → the bird gently lands on the lion"
                    ),
                    lines=4,
                )

                with gr.Row():
                    nlp_extract_btn  = gr.Button("🧬 Extraer acciones", variant="primary")
                    nlp_validate_btn = gr.Button("✅ Validar secuencia completa")
                    nlp_enhance_btn  = gr.Button("✨ Enriquecer prompt")

                nlp_actions_out  = gr.Markdown(label="Acciones detectadas")
                nlp_validate_out = gr.Markdown(label="Reporte de validación")
                nlp_enhanced_out = gr.Textbox(label="Prompt enriquecido", lines=3, interactive=False)

                gr.Markdown("---")
                gr.Markdown("""
#### Sobre la validación v2

El validador ahora corre **tres capas** en paralelo:

| Capa | Qué detecta | Score |
|---|---|---|
| 🧠 Semántica | Transiciones imposibles, exclusiones, pre-requisitos | 45% |
| 🏃 Momentum | Paradas bruscas, falta de build-up, cambios de velocidad | 25% |
| ⚙️ Física | Aceleraciones, ground-penetration, jitter de cámara | 30% |

El **score compuesto** va de 0 (incoherente) a 1.0 (perfecto).
                """)

            # ══════════════════════════════════════════════════════════════
            # TAB 6 — LoRA RECOMMENDER
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🤖 LoRA Recomendador"):
                gr.Markdown("""
### 🤖 Recomendador Automático de LoRAs
Analiza tu prompt con el **ActionExtractor**, detecta acciones y estilo, y sugiere
las LoRAs más relevantes con su rango de frames y curva de intensidad sugeridos.
                """)

                with gr.Row():
                    rec_prompt_in = gr.Textbox(
                        label="Prompt",
                        placeholder="A warrior runs and jumps in slow motion, cinematic...",
                        lines=3,
                        scale=3,
                    )
                    with gr.Column(scale=1):
                        rec_model_sel = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value=list(MODELS.keys())[0],
                            label="Modelo base",
                        )
                        rec_frames_in = gr.Number(49, label="Frames totales", precision=0)

                with gr.Row():
                    rec_analyze_btn = gr.Button("🔍 Analizar y recomendar", variant="primary")
                    rec_apply_btn   = gr.Button("⚡ Aplicar al schedule JSON", variant="secondary")

                rec_out_md     = gr.Markdown(label="Recomendaciones")
                rec_json_out   = gr.Textbox(
                    label="Schedule JSON generado (copia en pestaña LoRAs)",
                    lines=12, interactive=False,
                )

                gr.Markdown("""
---
#### Cómo funciona el scoring

```
Relevancia de acción   0–0.50   (¿La LoRA maneja esta acción?)
Relevancia de estilo   0–0.30   (¿Coincide con el estilo del prompt?)
Popularidad histórica  0–0.20   (¿Se usó con éxito antes?)
Trigger word bonus     0–0.10   (¿El prompt activa el trigger?)
────────────────────────────────
Score total            0–1.00
```
Compatibilidad de modelo se aplica como penalización si hay incompatibilidad.
                """)

            # ══════════════════════════════════════════════════════════════
            # TAB 7 — SKELETAL ANIMATOR
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🦴 Esqueleto"):
                gr.Markdown("""
### 🦴 Skeletal Animator
Construye secuencias de animación desde **poses nombradas** y genera los tokens de prompt
correspondientes en cada frame. Soporta esqueletos: `human`, `quadruped`, `bird`.
                """)

                with gr.Row():
                    skel_type = gr.Dropdown(
                        choices=list(SKELETON_TEMPLATES.keys()),
                        value="human",
                        label="Tipo de esqueleto",
                    )
                    skel_frames_in = gr.Number(49, label="Frames", precision=0)

                skel_poses_list_btn = gr.Button("📋 Ver poses disponibles", size="sm")
                skel_poses_available = gr.Markdown()

                gr.Markdown("#### Interpolación entre dos poses")
                with gr.Row():
                    skel_pose_a = gr.Dropdown(
                        choices=list(HUMAN_POSES_NAMES := list(POSE_LIBRARY.get("human", {}).keys())),
                        value=HUMAN_POSES_NAMES[0] if HUMAN_POSES_NAMES else "idle",
                        label="Pose A (inicio)",
                    )
                    skel_pose_b = gr.Dropdown(
                        choices=HUMAN_POSES_NAMES,
                        value=HUMAN_POSES_NAMES[1] if len(HUMAN_POSES_NAMES) > 1 else "walking",
                        label="Pose B (fin)",
                    )
                    skel_t = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="t (0=A, 1=B)")

                skel_preview_btn = gr.Button("👁️ Preview pose interpolada", size="sm")
                skel_preview_out = gr.Markdown()

                gr.Markdown("#### Secuencia multi-pose")
                skel_seq_input = gr.Textbox(
                    label="Poses en orden (separadas por comas)",
                    placeholder="idle, walking, running, jumping_peak, crouching, idle",
                    lines=2,
                )
                with gr.Row():
                    skel_seq_btn      = gr.Button("📽️ Generar secuencia", variant="primary")
                    skel_prompt_btn   = gr.Button("📝 → Copiar prompt en t=0.5")
                skel_seq_out    = gr.Markdown()
                skel_prompt_out = gr.Textbox(label="Prompt copiado", lines=2, interactive=False)

                gr.Markdown("""
---
#### Referencia de poses

| Esqueleto | Poses disponibles |
|---|---|
| `human` | idle, walking, running, jumping_peak, crouching, sitting, lying_down, pointing, waving, arms_raised, bowing, dancing |
| `quadruped` | idle, walking, running, roaring, jumping, crouching |
| `bird` | perched, flying, wings_up, wings_down, landing |
                """)

            # ══════════════════════════════════════════════════════════════
            # TAB 8 — GALLERY & A/B COMPARISON
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("🖼️ Galería y Comparación"):
                gr.Markdown("### Galería de Videos Generados")
                gallery_videos = gr.Gallery(label="Galería", format="mp4", columns=3, height="auto")
                refresh_gallery_btn = gr.Button("🔄 Refrescar Galería")
                
                gr.Markdown("---")
                gr.Markdown("### Comparación A/B")
                with gr.Row():
                    with gr.Column():
                        cmp_gen_1 = gr.Dropdown(label="Generación 1 (ID)", choices=[])
                        vid_1 = gr.Video(label="Video 1")
                        prompt_1 = gr.Markdown()
                    with gr.Column():
                        cmp_gen_2 = gr.Dropdown(label="Generación 2 (ID)", choices=[])
                        vid_2 = gr.Video(label="Video 2")
                        prompt_2 = gr.Markdown()
                
                with gr.Row():
                    cmp_title = gr.Textbox(label="Título para la comparación")
                    save_cmp_btn = gr.Button("💾 Guardar Comparación")
                cmp_status = gr.Markdown()

                def populate_gallery():
                    history = _db.get_history(limit=50)
                    items = []
                    choices = []
                    for r in history:
                        if Path(r.output_path).exists():
                            items.append((r.output_path, f"ID {r.id}: {r.prompt[:30]}..."))
                            choices.append(f"{r.id} - {r.model}")
                    return items, gr.update(choices=choices), gr.update(choices=choices)

                refresh_gallery_btn.click(
                    populate_gallery, outputs=[gallery_videos, cmp_gen_1, cmp_gen_2]
                )
                
                def load_cmp_video(choice):
                    if not choice: return None, ""
                    gen_id = int(choice.split(" ")[0])
                    rec = _db.get_generation(gen_id)
                    if not rec: return None, ""
                    return rec.output_path, f"**Prompt:** {rec.prompt}"
                    
                cmp_gen_1.change(load_cmp_video, inputs=cmp_gen_1, outputs=[vid_1, prompt_1])
                cmp_gen_2.change(load_cmp_video, inputs=cmp_gen_2, outputs=[vid_2, prompt_2])
                
                def save_comparison(title, c1, c2):
                    if not title or not c1 or not c2: return "⚠️ Faltan datos"
                    id1 = int(c1.split(" ")[0])
                    id2 = int(c2.split(" ")[0])
                    _db.save_comparison(title, [id1, id2])
                    return "✅ Comparación guardada exitosamente"
                    
                save_cmp_btn.click(save_comparison, inputs=[cmp_title, cmp_gen_1, cmp_gen_2], outputs=cmp_status)

                # Initialize gallery on load
                demo.load(populate_gallery, outputs=[gallery_videos, cmp_gen_1, cmp_gen_2])

            # ══════════════════════════════════════════════════════════════
            # TAB 9 — STORY MODE
            # ══════════════════════════════════════════════════════════════
            with gr.Tab("📖 Story Mode"):
                gr.Markdown("""
### 📖 Story Mode — Genera historias completas con continuidad visual

Escribe un **guión** con una escena por línea. El sistema generará automáticamente:
1. **Escena 1** usando Text-to-Video (CogVideoX-5B), o desde tu imagen de referencia con I2V.
2. **Escenas 2+** usando el **último frame de la escena anterior** como input de I2V, 
   garantizando que los personajes, colores y entorno se mantengan consistentes.
3. **Crossfade** suave entre todas las escenas para transiciones cinematográficas.

> **💡 Tip:** Si quieres que tu personaje se vea EXACTAMENTE igual en todas las escenas, 
> sube una **imagen de referencia** del personaje. La primera escena la generará a partir 
> de esa imagen, y cada escena siguiente heredará la apariencia visual.
                """)

                with gr.Row(equal_height=False):
                    # ── LEFT: Script & Settings ───────────────────────────
                    with gr.Column(scale=1, min_width=420):
                        story_script = gr.Textbox(
                            label="📝 Guión (una escena por línea)",
                            placeholder=(
                                "# Ejemplo: La Princesa y el Dragón\n"
                                "1. A beautiful princess with long silver hair walks through a dark enchanted forest at twilight\n"
                                "2. She discovers a hidden cave entrance glowing with blue crystals\n"
                                "3. Inside the cave, a massive friendly dragon sleeps peacefully\n"
                                "4. The dragon opens its golden eyes and looks at the princess curiously\n"
                                "5. The princess smiles and gently touches the dragon's nose\n"
                                "6. The dragon and princess fly together above the clouds at sunset"
                            ),
                            lines=12,
                        )

                        with gr.Accordion("🖼️ Imagen de Referencia (opcional)", open=False):
                            story_ref_image = gr.Image(
                                label="Imagen del personaje principal",
                                type="numpy",
                            )
                            gr.Markdown(
                                "Sube una imagen del personaje principal. La primera escena "
                                "se generará a partir de esta imagen usando I2V, asegurando "
                                "que su apariencia sea consistente en toda la historia.\n\n"
                                "**Sin imagen:** La escena 1 se genera con T2V (texto puro)."
                            )

                        with gr.Accordion("✨ Estilo Cinematográfico", open=False):
                            with gr.Row():
                                story_lighting = gr.Dropdown(
                                    LIGHTING_OPTIONS, value="—", label="💡 Iluminación")
                                story_cinematography = gr.Dropdown(
                                    CINEMATOGRAPHY_OPTIONS, value="—", label="🎥 Cinematografía")
                            with gr.Row():
                                story_quality = gr.Dropdown(
                                    QUALITY_OPTIONS, value="—", label="⭐ Calidad")
                                story_atmosphere = gr.Dropdown(
                                    ATMOSPHERE_OPTIONS, value="—", label="🌫️ Atmósfera")
                            with gr.Row():
                                story_genre = gr.Dropdown(
                                    GENRE_OPTIONS, value="—", label="🎭 Género")
                                story_negative = gr.Dropdown(
                                    NEGATIVE_PRESETS_LIST, value="standard", label="🚫 Negative")

                        with gr.Accordion("⚙️ Parámetros de Generación", open=False):
                            with gr.Row():
                                story_frames = gr.Slider(
                                    8, 49, 25, step=8,
                                    label="Frames por escena",
                                )
                                story_fps = gr.Slider(8, 24, 12, step=1, label="FPS")
                            with gr.Row():
                                story_guidance = gr.Slider(
                                    1.0, 15.0, 6.0, step=0.5, label="Guidance")
                                story_steps = gr.Slider(
                                    10, 50, 30, step=5, label="Pasos por escena")
                            story_seed = gr.Number(
                                -1, label="Seed (-1 = aleatorio)", precision=0)
                            story_crossfade = gr.Slider(
                                0, 12, 4, step=1,
                                label="🔀 Frames de crossfade entre escenas")

                        story_gen_btn = gr.Button(
                            "📖 Generar Historia Completa",
                            variant="primary",
                            elem_classes="gen-btn",
                        )

                        gr.Markdown("""
---
#### ⏱️ Estimación de tiempo
| Escenas | Frames/escena | FPS | Duración aprox. | Tiempo GPU (RTX 3090) |
|---------|--------------|-----|-----------------|----------------------|
| 3       | 25           | 12  | ~6 seg          | ~3-5 min             |
| 5       | 25           | 12  | ~10 seg         | ~5-8 min             |
| 8       | 25           | 12  | ~16 seg         | ~8-13 min            |
| 5       | 49           | 24  | ~10 seg         | ~10-15 min           |
| 10      | 25           | 12  | ~20 seg         | ~15-20 min           |

> **Nota:** La primera escena usa CogVideoX-5B T2V (~14 GB). Las siguientes 
> usan CogVideoX-5B-I2V (~14 GB). El sistema cambiará entre modelos automáticamente.
                        """)

                    # ── RIGHT: Output ────────────────────────────────────
                    with gr.Column(scale=1, min_width=400):
                        story_output_video = gr.Video(label="🎥 Historia Completa")
                        story_output_info = gr.Markdown()

        # ──────────────────────────────────────────────────────────────────
        # Event wiring
        # ──────────────────────────────────────────────────────────────────

        # Model badge
        def _mode_badge(key):
            mtype = MODELS[key]["type"]
            if mtype == "v2v":
                return (
                    gr.update(visible=False),   # input_image
                    "**Modo:** Video → Video",  # mode_badge
                    gr.update(visible=True),    # input_video
                    gr.update(visible=True),    # v2v_strength
                )
            elif mtype == "i2v":
                return (
                    gr.update(visible=True),
                    "**Modo:** Imagen → Video",
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
            else:
                return (
                    gr.update(visible=False),
                    "**Modo:** Texto → Video",
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
        model_key.change(
            _mode_badge, model_key,
            [input_image, mode_badge, input_video, v2v_strength],
        )

        # Gesture preview
        def _gesture_preview(name, t):
            if name == "—":
                return ""
            p = gesture_to_prompt(name, t)
            return f"**Preview prompt:** `{p}`"
        gesture_name.change(_gesture_preview, [gesture_name, gesture_t], gesture_preview)
        gesture_t.change(_gesture_preview, [gesture_name, gesture_t], gesture_preview)

        # Validate
        validate_btn.click(
            lambda p, _: _validator.full_analysis(p),
            inputs=[raw_prompt, num_frames],
            outputs=validation_md,
        )

        # Generate
        gen_btn.click(
            generate_video,
            inputs=[
                model_key, raw_prompt, negative_preset,
                lighting, cinematography, quality, atmosphere, genre,
                camera_preset,
                gesture_name, gesture_t,
                active_loras, lora_schedule_json,
                num_frames, fps, guidance, steps, width, height, seed,
                input_image, input_video, v2v_strength,
                crossfade_frames,
            ],
            outputs=[output_video, output_info, val_out_md],
        )

        # Story Mode
        story_gen_btn.click(
            generate_story,
            inputs=[
                story_script, story_negative,
                story_lighting, story_cinematography,
                story_quality, story_atmosphere, story_genre,
                story_frames, story_fps, story_guidance,
                story_steps, story_seed, story_crossfade,
                story_ref_image,
            ],
            outputs=[story_output_video, story_output_info],
        )

        # LoRA manager
        add_lora_btn.click(
            add_lora,
            inputs=[l_name, l_source, l_cat, l_scale, l_wname,
                    l_fstart, l_fend, l_curve],
            outputs=[lora_add_status, lora_list, active_loras],
        )
        remove_lora_btn.click(
            remove_loras,
            inputs=lora_list,
            outputs=[lora_rm_status, lora_list, active_loras],
        )
        refresh_lora_btn.click(refresh_loras, outputs=[lora_list, active_loras])

        # Presets
        save_preset_btn.click(
            save_preset_fn,
            inputs=[preset_name_in, preset_cat_in, model_key, raw_prompt,
                    negative_preset, lighting, cinematography,
                    quality, atmosphere, genre, preset_notes_in],
            outputs=preset_status,
        )
        load_preset_btn.click(
            load_preset_fn,
            inputs=preset_sel,
            outputs=[raw_prompt, negative_preset, lighting,
                     cinematography, quality, atmosphere, genre, preset_notes_in],
        )

        # History
        refresh_hist_btn.click(get_history_table, outputs=history_table)
        hist_del_btn.click(delete_gen, inputs=hist_del_id,
                           outputs=[hist_status, history_table])
        stats_btn.click(
            lambda: (
                "**Estadísticas:**\n" +
                "\n".join(f"- {k}: {v}" for k, v in _db.stats().items())
            ),
            outputs=stats_md,
        )

        # ── TAB 5: NLP Analysis ───────────────────────────────────────────
        nlp_extract_btn.click(
            extract_actions_fn,
            inputs=nlp_prompt_in,
            outputs=nlp_actions_out,
        )
        nlp_validate_btn.click(
            lambda p: _validator.full_analysis(p),
            inputs=nlp_prompt_in,
            outputs=nlp_validate_out,
        )
        nlp_enhance_btn.click(
            enhance_from_actions_fn,
            inputs=nlp_prompt_in,
            outputs=nlp_enhanced_out,
        )

        # ── TAB 6: LoRA Recommender ───────────────────────────────────────
        rec_analyze_btn.click(
            recommend_loras_fn,
            inputs=[rec_prompt_in, rec_model_sel, rec_frames_in],
            outputs=rec_out_md,
        )
        rec_apply_btn.click(
            apply_recommended_loras_fn,
            inputs=[rec_prompt_in, rec_model_sel, rec_frames_in],
            outputs=[rec_out_md, rec_json_out],
        )

        # ── TAB 7: Skeletal Animator ──────────────────────────────────────
        def _update_pose_dropdowns(skel_type):
            poses = list(POSE_LIBRARY.get(skel_type, {}).keys())
            val_a = poses[0] if poses else ""
            val_b = poses[1] if len(poses) > 1 else val_a
            return (
                gr.update(choices=poses, value=val_a),
                gr.update(choices=poses, value=val_b),
            )
        skel_type.change(
            _update_pose_dropdowns,
            inputs=skel_type,
            outputs=[skel_pose_a, skel_pose_b],
        )
        skel_poses_list_btn.click(
            list_skeleton_poses_fn,
            inputs=skel_type,
            outputs=skel_poses_available,
        )
        skel_preview_btn.click(
            skeleton_preview_fn,
            inputs=[skel_type, skel_pose_a, skel_pose_b, skel_t],
            outputs=skel_preview_out,
        )
        skel_t.change(
            skeleton_preview_fn,
            inputs=[skel_type, skel_pose_a, skel_pose_b, skel_t],
            outputs=skel_preview_out,
        )
        skel_seq_btn.click(
            skeleton_sequence_fn,
            inputs=[skel_type, skel_seq_input, skel_frames_in],
            outputs=skel_seq_out,
        )
        skel_prompt_btn.click(
            lambda st, poses, t: skeleton_to_prompt_fn(st, poses, 0.5),
            inputs=[skel_type, skel_seq_input, skel_t],
            outputs=skel_prompt_out,
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host",  default="0.0.0.0")
    ap.add_argument("--port",  type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--auth",  nargs=2, metavar=("USER", "PASS"))
    args = ap.parse_args()

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        auth=tuple(args.auth) if args.auth else None,
        show_error=True,
    )
