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

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace token (set via: export HF_TOKEN=hf_xxxxx  in Linux/Vast.ai)
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    print(f"[INFO] HF_TOKEN encontrado (longitud={len(HF_TOKEN)}) — se usará para descargar modelos.")
else:
    print("[WARN] HF_TOKEN no definido. Los modelos con gating pueden fallar.\n"
          "  Para activarlo en Vast.ai: export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("outputs");  OUTPUT_DIR.mkdir(exist_ok=True)
LORAS_DIR   = Path("loras");    LORAS_DIR.mkdir(exist_ok=True)
LORAS_INDEX = LORAS_DIR / "index.json"

MODELS = {
    "CogVideoX-5B (T2V)": {
        "repo":     "THUDM/CogVideoX-5b",
        "type":     "t2v",
        "pipeline": "CogVideoXPipeline",
    },
    "CogVideoX-5B-I2V (Img→Vid)": {
        "repo":     "THUDM/CogVideoX-5b-I2V",
        "type":     "i2v",
        "pipeline": "CogVideoXImageToVideoPipeline",
    },
    "LTX-Video (T2V rápido)": {
        "repo":     "Lightricks/LTX-Video",
        "type":     "t2v",
        "pipeline": "LTXPipeline",
    },
    "LTX-Video-I2V (Img→Vid rápido)": {
        "repo":     "Lightricks/LTX-Video",
        "type":     "i2v",
        "pipeline": "LTXImageToVideoPipeline",
    },
}

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

_pipeline_cache: dict = {}
_current_model_key: str | None = None


def _load_lora_index() -> list:
    if LORAS_INDEX.exists():
        try:
            return json.loads(LORAS_INDEX.read_text())
        except Exception:
            return []
    return []

def _save_lora_index(idx: list):
    LORAS_INDEX.write_text(json.dumps(idx, indent=2, ensure_ascii=False))

def _lora_choices(idx: list) -> list:
    return [f"{l['name']}  [{l['category']}]" for l in idx]

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline management
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline(model_key: str, progress=None):
    global _pipeline_cache, _current_model_key
    if model_key == _current_model_key and model_key in _pipeline_cache:
        return _pipeline_cache[model_key]

    for v in list(_pipeline_cache.values()):
        del v
    torch.cuda.empty_cache()
    _pipeline_cache = {}

    cfg   = MODELS[model_key]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    from diffusers import (
        CogVideoXPipeline, CogVideoXImageToVideoPipeline,
        LTXPipeline, LTXImageToVideoPipeline,
    )
    cls_map = {
        "CogVideoXPipeline":             CogVideoXPipeline,
        "CogVideoXImageToVideoPipeline": CogVideoXImageToVideoPipeline,
        "LTXPipeline":                   LTXPipeline,
        "LTXImageToVideoPipeline":       LTXImageToVideoPipeline,
    }
    
    msg_desc = f"Descargando / Cargando {cfg['pipeline']} (~14GB - Tomará un momento)..."
    print(f"[INFO] {msg_desc}")
    if progress:
        progress(0.10, desc=msg_desc)
        
    load_kwargs = {"torch_dtype": dtype}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    pipe = cls_map[cfg["pipeline"]].from_pretrained(cfg["repo"], **load_kwargs)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    _pipeline_cache[model_key] = pipe
    _current_model_key = model_key
    return pipe

# ─────────────────────────────────────────────────────────────────────────────
# Frames → MP4
# ─────────────────────────────────────────────────────────────────────────────

def _frames_to_mp4(frames, fps: int = 24) -> str:
    import imageio
    out = str(OUTPUT_DIR / f"{uuid.uuid4().hex}.mp4")
    w = imageio.get_writer(out, fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p")
    for f in frames:
        if isinstance(f, Image.Image):
            f = np.array(f)
        w.append_data(f)
    w.close()
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Core generation logic
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipe(pipe, mode, prompt, negative, input_image,
              num_frames, guidance, steps, width, height, gen, progress=None, base_prog=0.0):
    
    def step_callback(pipe_cls, step: int, timestep: int, callback_kwargs: dict):
        if progress is not None:
            progress(base_prog, desc=f"⏳ Inferencia de IA: Procesando Paso {step+1}/{steps}...")
        return callback_kwargs

    if mode == "t2v":
        # Force dimensions to be divisible by 32 for LTX
        w_t2v = (width // 32) * 32
        h_t2v = (height // 32) * 32
        return pipe(
            prompt=prompt, negative_prompt=negative or None,
            num_frames=num_frames, guidance_scale=guidance,
            num_inference_steps=steps,
            width=w_t2v, height=h_t2v, generator=gen,
            callback_on_step_end=step_callback,
        ).frames[0]
    else:
        img = Image.fromarray(input_image).convert("RGB")
        is_cogvideox_i2v = pipe.__class__.__name__ == "CogVideoXImageToVideoPipeline"
        
        if is_cogvideox_i2v:
            # CogVideoX-5B-I2V strictly requires exactly 720x480 image and fails if width/height args are explicitly passed.
            img = img.resize((720, 480), Image.LANCZOS)
            return pipe(
                image=img, prompt=prompt, negative_prompt=negative or None,
                num_frames=num_frames, guidance_scale=guidance,
                num_inference_steps=steps, generator=gen,
                callback_on_step_end=step_callback,
            ).frames[0]
        else:
            w, h = img.size
            # For LTX, ensure the scaled dimensions are strictly divisible by 32.
            sc = 720 / max(w, h)
            w_scaled = (int(w * sc) // 32) * 32
            h_scaled = (int(h * sc) // 32) * 32
            
            # Prevent 0 dimension error if image aspect ratio is extreme
            w_scaled = max(32, w_scaled)
            h_scaled = max(32, h_scaled)
            
            img = img.resize((w_scaled, h_scaled), Image.LANCZOS)
            return pipe(
                image=img, prompt=prompt, negative_prompt=negative or None,
                num_frames=num_frames, guidance_scale=guidance,
                num_inference_steps=steps, generator=gen,
                height=h_scaled, width=w_scaled,
                callback_on_step_end=step_callback,
            ).frames[0]

# ─────────────────────────────────────────────────────────────────────────────
# SceneGraph helpers
# ─────────────────────────────────────────────────────────────────────────────

_POSITION_MAP = {
    "left":         (0.2, 0.5, 0.0),
    "left frame":   (0.15, 0.5, 0.0),
    "right":        (0.8, 0.5, 0.0),
    "right frame":  (0.85, 0.5, 0.0),
    "center":       (0.5, 0.5, 0.0),
    "top":          (0.5, 0.2, 0.0),
    "bottom":       (0.5, 0.8, 0.0),
    "foreground":   (0.5, 0.6, -0.3),
    "background":   (0.5, 0.4, 0.5),
}

def _position_hint_to_coords(hint: str | None) -> tuple:
    """Convert a textual position hint to (x, y, z) coords."""
    if not hint:
        return (0.5, 0.5, 0.0)
    return _POSITION_MAP.get(hint.lower().strip(), (0.5, 0.5, 0.0))


def _register_interaction(
    graph: SceneGraph,
    interaction_desc: str,
    kf,
    total_frames: int,
):
    """
    Parse a natural language interaction string and register it in the SceneGraph.
    E.g. "Bird lands on Lion's head" → subject=CHAR_2, object=CHAR_1, relation=on_top_of
    """
    import re
    desc_lower = interaction_desc.lower()
    # Try to match referenced characters by label
    obj_ids = list(graph.objects.keys())
    subject_id, object_id = None, None
    for oid, obj in graph.objects.items():
        label_low = obj.label.lower()
        idx = desc_lower.find(label_low)
        if idx != -1:
            if subject_id is None:
                subject_id = oid
            elif object_id is None and oid != subject_id:
                object_id = oid

    if not subject_id or not object_id:
        return  # Can't resolve both subjects

    # Detect relation type
    relation = "beside"  # default
    relation_hints = {
        "on top of":    "on_top_of",
        "lands on":     "on_top_of",
        "resting on":   "on_top_of",
        "sits on":      "on_top_of",
        "behind":       "behind",
        "in front of":  "in_front_of",
        "approaching":  "approaching",
        "moves toward": "approaching",
        "walks toward": "approaching",
        "leaving":      "leaving",
        "moves away":   "leaving",
        "touching":     "touching",
        "holds":        "holding",
        "holding":      "holding",
        "grabs":        "holding",
        "inside":       "inside",
        "surrounding":  "surrounding",
    }
    for hint, rel in relation_hints.items():
        if hint in desc_lower:
            relation = rel
            break

    frame_start = round(kf.t_start * total_frames)
    frame_end = round(kf.t_end * total_frames)
    graph.add_interaction(
        subject_id=subject_id,
        object_id=object_id,
        relation=relation,
        frame_start=frame_start,
        frame_end=frame_end,
        description=interaction_desc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_video(
    # Core
    model_key, raw_prompt, negative_preset,
    # Enhancement
    lighting, cinematography, quality, atmosphere, genre,
    # Motion (legacy dropdowns kept for quick use)
    camera_preset,
    # Gestures
    gesture_name, gesture_t,
    # LoRA
    active_lora_names,
    # LoRA schedule JSON (advanced)
    lora_schedule_json,
    # Generation params
    num_frames, fps, guidance, steps, width, height, seed,
    # Input image
    input_image,
    progress=gr.Progress(track_tqdm=False),
):
    if not raw_prompt.strip():
        raise gr.Error("⚠️ El prompt no puede estar vacío.")

    cfg  = MODELS[model_key]
    mode = cfg["type"]
    if mode == "i2v" and input_image is None:
        raise gr.Error("⚠️ Este modelo requiere imagen de entrada.")

    # ── 1. Scene parsing ──────────────────────────────────────────────────────
    progress(0.02, desc="📖 Parseando escenas…")
    keyframes = _parser.parse(raw_prompt, total_frames=num_frames)

    # ── 1b. Scene graph — multi-subject tracking & spatial relations ──────────
    scene_graph = SceneGraph()
    for kf in keyframes:
        for char in kf.characters:
            if char.char_id not in scene_graph.objects:
                pos = _position_hint_to_coords(char.position)
                scene_graph.add_object(
                    obj_id=char.char_id,
                    label=char.label,
                    category="personaje",
                    position=pos,
                )
            if char.action:
                scene_graph.set_action(char.char_id, char.action)
        # Register interactions from structured parsing
        for inter_desc in kf.interactions:
            _register_interaction(scene_graph, inter_desc, kf, num_frames)

    # ── 2. Temporal validation ────────────────────────────────────────────────
    validation = _validator.validate_keyframes(keyframes)
    val_report = validation.to_markdown()

    # ── 3. Prompt enhancement ─────────────────────────────────────────────────
    progress(0.05, desc="✍️ Mejorando prompts…")
    MAX_PROMPT_TOKENS = 200  # CogVideoX limit ~226 tokens
    enhanced_prompts = []
    for kf in keyframes:
        base = kf.description
        
        # Add SceneGraph spatial relationships & interactions
        scene_suffix = scene_graph.to_prompt_suffix(frame=round(kf.t_start * num_frames))
        if scene_suffix:
            base += f", {scene_suffix}"

        # Add camera path hint if available
        if kf.camera_hint:
            base += f", {kf.camera_hint}"
        # Add gesture
        g_prompt = gesture_to_prompt(gesture_name, gesture_t) if gesture_name != "—" else ""
        if g_prompt:
            base += f", {g_prompt}"
        # Camera preset prompt
        if camera_preset and camera_preset != "—":
            path = CameraPath.from_preset(camera_preset, num_frames)
            cam_desc = path.to_prompt_description(
                round(kf.t_start * num_frames), num_frames
            )
            if cam_desc:
                base += f", {cam_desc}"
        # Enhance
        enhanced = _enhancer.enhance(
            base=base,
            lighting=None if lighting == "—" else lighting,
            cinematography=None if cinematography == "—" else cinematography,
            quality=None if quality == "—" else quality,
            atmosphere=None if atmosphere == "—" else atmosphere,
            genre=None if genre == "—" else genre,
        )
        # Truncate if prompt exceeds token limit
        token_count = len(enhanced.split())
        if token_count > MAX_PROMPT_TOKENS:
            words = enhanced.split()[:MAX_PROMPT_TOKENS]
            enhanced = " ".join(words)
            gr.Warning(
                f"⚠️ Prompt truncado de {token_count} a {MAX_PROMPT_TOKENS} tokens "
                f"(keyframe {kf.index + 1}). Reduce mejoras cinematográficas."
            )
        enhanced_prompts.append(enhanced)

    # ── 4. Negative prompt ────────────────────────────────────────────────────
    negative = _enhancer.get_negative(negative_preset)

    # ── 5. Load model ─────────────────────────────────────────────────────────
    progress(0.08, desc="🤖 Localizando e Inicializando Motor IA…")
    pipe = _load_pipeline(model_key, progress=progress)

    # ── 6. LoRA schedule ──────────────────────────────────────────────────────
    progress(0.12, desc="🎨 Configurando LoRAs…")
    sched = DynamicLoRASchedule(pipe, total_frames=num_frames)

    # Static LoRAs from UI checkboxes
    lora_index = _load_lora_index()
    name_map   = {f"{l['name']}  [{l['category']}]": l for l in lora_index}
    for disp in (active_lora_names or []):
        entry = name_map.get(disp)
        if entry:
            sched.add(
                lora_id=entry["name"].replace(" ", "_").lower(),
                source=entry["source"],
                base_scale=float(entry.get("scale", 0.8)),
                weight_name=entry.get("weight_name"),
                category=entry.get("category", "estilo"),
                label=entry.get("name", disp),
            )
    # Advanced schedule from JSON textarea
    if lora_schedule_json and lora_schedule_json.strip():
        try:
            custom = json.loads(lora_schedule_json)
            sched.add_from_dict(custom)
        except json.JSONDecodeError as e:
            gr.Warning(f"⚠️ JSON de LoRA schedule inválido (se ignoró): {e}")

    # ── 7. Generation (per keyframe) ──────────────────────────────────────────
    gen = (torch.Generator("cuda").manual_seed(int(seed))
           if int(seed) != -1 else None)
    t0 = time.time()
    all_frames = []

    for i, (kf, enhanced) in enumerate(zip(keyframes, enhanced_prompts)):
        base_progress = 0.15 + 0.75 * (i / len(keyframes))
        progress(base_progress, desc=f"🎬 Preparando entorno de generacion {i+1}/{len(keyframes)}…")
        
        # Apply LoRA weights for this keyframe's midpoint frame
        mid_frame = round((kf.t_start + kf.t_end) / 2 * num_frames)
        sched.apply_for_frame(mid_frame)

        # Compute frames for this segment
        seg_frames = max(
            8,
            round((kf.t_end - kf.t_start) * num_frames)
        )
        # Make divisible by 8 + 1 for CogVideoX
        if "CogVideoX" in model_key:
            seg_frames = (seg_frames // 8) * 8 + 1

        try:
            frames = _run_pipe(
                pipe, mode, enhanced, negative, input_image,
                seg_frames, guidance, steps, width, height, gen,
                progress=progress, base_prog=base_progress
            )
            all_frames.extend(frames)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error("❌ GPU sin memoria. Reduce resolución o frames.")

    # ── 8. Export ─────────────────────────────────────────────────────────────
    progress(0.92, desc="💾 Exportando MP4…")
    out_path = _frames_to_mp4(all_frames, fps=fps)
    elapsed  = time.time() - t0

    # ── 9. Save to DB ─────────────────────────────────────────────────────────
    _db.save_generation(GenerationRecord(
        model=model_key,
        prompt=raw_prompt,
        negative=negative,
        params=dict(
            frames=num_frames, fps=fps, guidance=guidance,
            steps=steps, width=width, height=height, seed=int(seed),
        ),
        loras=active_lora_names or [],
        motion_tags=dict(
            camera=camera_preset, lighting=lighting,
            cinematography=cinematography, quality=quality,
            atmosphere=atmosphere, genre=genre,
        ),
        output_path=out_path,
        duration_s=elapsed,
        frame_count=len(all_frames),
    ))

    # ── 10. Build info markdown ───────────────────────────────────────────────
    kf_summary = "\n".join(
        f"  `{i+1}.` {p[:80]}…" if len(p) > 80 else f"  `{i+1}.` {p}"
        for i, p in enumerate(enhanced_prompts)
    )
    info = (
        f"✅ **{len(all_frames)} frames** en **{elapsed:.1f}s**\n\n"
        f"**Keyframes generados ({len(keyframes)}):**\n{kf_summary}\n\n"
        f"**Coherencia temporal:**\n{val_report}\n\n"
        f"📁 `{out_path}`"
    )
    return out_path, info, val_report

# ─────────────────────────────────────────────────────────────────────────────
# LoRA manager
# ─────────────────────────────────────────────────────────────────────────────

def add_lora(name, source, category, scale, weight_name, frame_start, frame_end, curve):
    name = name.strip(); source = source.strip()
    if not name or not source:
        return "⚠️ Nombre y fuente son obligatorios.", gr.update(), gr.update()
    idx = _load_lora_index()
    if any(l["name"] == name for l in idx):
        return f"⚠️ Ya existe '{name}'.", gr.update(), gr.update()
    idx.append({
        "name": name, "source": source, "category": category,
        "scale": round(float(scale), 2),
        "weight_name": weight_name.strip() or None,
        "frame_start": int(frame_start),
        "frame_end":   int(frame_end),
        "curve":       curve,
    })
    _save_lora_index(idx)
    choices = _lora_choices(idx)
    return f"✅ LoRA '{name}' añadida.", gr.update(choices=choices, value=[]), gr.update(choices=choices)

def remove_loras(selected):
    if not selected:
        return "⚠️ Selecciona al menos una.", gr.update(), gr.update()
    idx = _load_lora_index()
    idx = [l for l in idx if f"{l['name']}  [{l['category']}]" not in selected]
    _save_lora_index(idx)
    choices = _lora_choices(idx)
    return f"🗑️ {len(selected)} eliminada(s).", gr.update(choices=choices, value=[]), gr.update(choices=choices, value=[])

def refresh_loras():
    choices = _lora_choices(_load_lora_index())
    return gr.update(choices=choices), gr.update(choices=choices)

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
    user_loras = _load_lora_index()
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
    user_loras = _load_lora_index()
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
    lora_choices_init = _lora_choices(_load_lora_index())
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

        # ──────────────────────────────────────────────────────────────────
        # Event wiring
        # ──────────────────────────────────────────────────────────────────

        # Model badge
        def _mode_badge(key):
            is_i2v = MODELS[key]["type"] == "i2v"
            return (gr.update(visible=is_i2v),
                    "**Modo:** Imagen → Video" if is_i2v else "**Modo:** Texto → Video")
        model_key.change(_mode_badge, model_key, [input_image, mode_badge])

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
                input_image,
            ],
            outputs=[output_video, output_info, val_out_md],
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
