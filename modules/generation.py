"""
modules/generation.py
───────────────────────
Lógica principal de generación de un solo video (T2V, I2V, V2V).
"""
import time
import json
import torch
import numpy as np
import gradio as gr
from PIL import Image

from modules.scene_graph import SceneGraph
from modules.motion_interpolator import CameraPath
from modules.lora_scheduler import DynamicLoRASchedule
from modules.gesture_templates import gesture_to_prompt
from modules.database import GenerationRecord
from modules.pipeline_utils import load_pipeline, frames_to_mp4, extract_video_frames, _normalize_frame
from modules.cache_manager import segment_cache

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
    if not hint:
        return (0.5, 0.5, 0.0)
    return _POSITION_MAP.get(hint.lower().strip(), (0.5, 0.5, 0.0))


def _register_interaction(graph: SceneGraph, interaction_desc: str, kf, total_frames: int):
    desc_lower = interaction_desc.lower()
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
        return

    relation = "beside"
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


def crossfade_segments(segments: list[list], overlap_frames: int = 4) -> list:
    segments = [s for s in segments if s]
    if not segments:
        return []
    if len(segments) <= 1:
        return segments[0] if segments else []

    overlap_frames = max(1, overlap_frames)
    result = list(segments[0])

    for seg in segments[1:]:
        actual_overlap = min(overlap_frames, len(result), len(seg))
        if actual_overlap <= 0:
            result.extend(seg)
            continue

        blended = []
        for j in range(actual_overlap):
            alpha = (j + 1) / (actual_overlap + 1)
            frame_a = result[-(actual_overlap - j)]
            frame_b = seg[j]
            arr_a = np.array(frame_a) if isinstance(frame_a, Image.Image) else frame_a
            arr_b = np.array(frame_b) if isinstance(frame_b, Image.Image) else frame_b
            if arr_a.shape != arr_b.shape:
                frame_b_pil = Image.fromarray(arr_b).resize((arr_a.shape[1], arr_a.shape[0]), Image.LANCZOS)
                arr_b = np.array(frame_b_pil)
            mixed = ((1 - alpha) * arr_a.astype(np.float32) + alpha * arr_b.astype(np.float32))
            blended.append(mixed.astype(np.uint8))

        result = result[:-(actual_overlap)] + blended + list(seg[actual_overlap:])
    return result


def _run_pipe(pipe, mode, prompt, negative, input_image,
              num_frames, guidance, steps, width, height, gen,
              progress=None, base_prog=0.0, input_video_path=None,
              v2v_strength=0.7):
    def step_callback(pipe_cls, step: int, timestep: int, callback_kwargs: dict):
        if progress is not None:
            progress(base_prog, desc=f"⏳ Inferencia de IA: Procesando Paso {step+1}/{steps}...")
        return callback_kwargs

    if mode == "v2v":
        if not input_video_path:
            raise ValueError("El modo V2V requiere un video de entrada.")
        source_frames = extract_video_frames(input_video_path, max_frames=num_frames)
        if not source_frames:
            raise ValueError("No se pudieron extraer frames del video.")
        source_frames = [f.resize((720, 480), Image.LANCZOS) for f in source_frames]
        raw_frames = pipe(
            video=source_frames,
            prompt=prompt, negative_prompt=negative or None,
            num_frames=len(source_frames),
            guidance_scale=guidance,
            strength=v2v_strength,
            num_inference_steps=steps, generator=gen,
            callback_on_step_end=step_callback,
        ).frames[0]

    elif mode == "t2v":
        w_t2v = (width // 32) * 32
        h_t2v = (height // 32) * 32
        raw_frames = pipe(
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
            img = img.resize((720, 480), Image.LANCZOS)
            raw_frames = pipe(
                image=img, prompt=prompt, negative_prompt=negative or None,
                num_frames=num_frames, guidance_scale=guidance,
                num_inference_steps=steps, generator=gen,
                callback_on_step_end=step_callback,
            ).frames[0]
        else:
            w, h = img.size
            sc = 720 / max(w, h)
            w_scaled = max(32, (int(w * sc) // 32) * 32)
            h_scaled = max(32, (int(h * sc) // 32) * 32)
            img = img.resize((w_scaled, h_scaled), Image.LANCZOS)
            raw_frames = pipe(
                image=img, prompt=prompt, negative_prompt=negative or None,
                num_frames=num_frames, guidance_scale=guidance,
                num_inference_steps=steps, generator=gen,
                height=h_scaled, width=w_scaled,
                callback_on_step_end=step_callback,
            ).frames[0]

    # ── CRITICAL: Normalize all frames to uint8 PIL Images ──────────────────
    # The diffusers pipeline may return frames as float tensors [0,1], [-1,1],
    # or PIL Images depending on the pipeline version and post-processor.
    # Without normalization, downstream code (MP4 export, crossfade) may receive
    # corrupt data, producing garbled or black outputs.
    normalized = []
    for f in raw_frames:
        arr = _normalize_frame(f)
        normalized.append(Image.fromarray(arr))
    return normalized



def generate_video(
    model_key, raw_prompt, negative_preset,
    lighting, cinematography, quality, atmosphere, genre,
    camera_preset, gesture_name, gesture_t,
    active_lora_names, lora_schedule_json,
    num_frames, fps, guidance, steps, width, height, seed,
    input_image, input_video, v2v_strength, crossfade_frames,
    # Injected singletons/callbacks:
    deps: dict,
    progress=gr.Progress(track_tqdm=False)
):
    if not raw_prompt.strip():
        raise gr.Error("⚠️ El prompt no puede estar vacío.")

    cfg  = deps["MODELS"][model_key]
    mode = cfg["type"]
    if mode == "i2v" and input_image is None:
        raise gr.Error("⚠️ Este modelo requiere imagen de entrada.")
    if mode == "v2v" and input_video is None:
        raise gr.Error("⚠️ Este modelo requiere un video de entrada.")

    input_video_path = None
    if mode == "v2v" and input_video is not None:
        input_video_path = input_video if isinstance(input_video, str) else input_video.name

    progress(0.02, desc="📖 Parseando escenas…")
    keyframes = deps["parser"].parse(raw_prompt, total_frames=num_frames)

    scene_graph = SceneGraph()
    for kf in keyframes:
        for char in kf.characters:
            if char.char_id not in scene_graph.objects:
                pos = _position_hint_to_coords(char.position)
                scene_graph.add_object(
                    obj_id=char.char_id, label=char.label,
                    category="personaje", position=pos,
                )
            if char.action:
                scene_graph.set_action(char.char_id, char.action)
        for inter_desc in kf.interactions:
            _register_interaction(scene_graph, inter_desc, kf, num_frames)

    validation = deps["validator"].validate_keyframes(keyframes)
    val_report = validation.to_markdown()

    progress(0.05, desc="✍️ Mejorando prompts…")
    MAX_PROMPT_TOKENS = 200
    enhanced_prompts = []
    for kf in keyframes:
        base = kf.description
        scene_suffix = scene_graph.to_prompt_suffix(frame=round(kf.t_start * num_frames))
        if scene_suffix:
            base += f", {scene_suffix}"
        if kf.camera_hint:
            base += f", {kf.camera_hint}"
        g_prompt = gesture_to_prompt(gesture_name, gesture_t) if gesture_name != "—" else ""
        if g_prompt:
            base += f", {g_prompt}"
        if camera_preset and camera_preset != "—":
            path = CameraPath.from_preset(camera_preset, num_frames)
            cam_desc = path.to_prompt_description(round(kf.t_start * num_frames), num_frames)
            if cam_desc:
                base += f", {cam_desc}"
        
        enhanced = deps["enhancer"].enhance(
            base=base,
            lighting=None if lighting == "—" else lighting,
            cinematography=None if cinematography == "—" else cinematography,
            quality=None if quality == "—" else quality,
            atmosphere=None if atmosphere == "—" else atmosphere,
            genre=None if genre == "—" else genre,
        )
        token_count = len(enhanced.split())
        if token_count > MAX_PROMPT_TOKENS:
            words = enhanced.split()[:MAX_PROMPT_TOKENS]
            enhanced = " ".join(words)
            gr.Warning(f"⚠️ Prompt truncado de {token_count} a {MAX_PROMPT_TOKENS} tokens (keyframe {kf.index + 1}).")
        enhanced_prompts.append(enhanced)

    negative = deps["enhancer"].get_negative(negative_preset)

    progress(0.08, desc="🤖 Localizando e Inicializando Motor IA…")
    pipe = load_pipeline(model_key, progress=progress)

    progress(0.12, desc="🎨 Configurando LoRAs…")
    sched = DynamicLoRASchedule(pipe, total_frames=num_frames)

    lora_index = deps["load_lora_index"]()
    name_map   = {f"{l['name']}  [{l['category']}]": l for l in lora_index}
    for disp in (active_lora_names or []):
        entry = name_map.get(disp)
        if entry:
            sched.add(
                lora_id=entry["name"].replace(" ", "_").lower(),
                source=entry["source"], base_scale=float(entry.get("scale", 0.8)),
                weight_name=entry.get("weight_name"), category=entry.get("category", "estilo"),
                label=entry.get("name", disp),
            )
            
    if lora_schedule_json and lora_schedule_json.strip():
        try:
            custom = json.loads(lora_schedule_json)
            sched.add_from_dict(custom)
        except json.JSONDecodeError as e:
            gr.Warning(f"⚠️ JSON de LoRA schedule inválido (se ignoró): {e}")

    gen = (torch.Generator("cuda").manual_seed(int(seed)) if int(seed) != -1 else None)
    
    progress(0.14, desc="📦 Precargando adaptadores LoRA en memoria…")
    sched.preload_all()
    
    t0 = time.time()
    segments = []

    for i, (kf, enhanced) in enumerate(zip(keyframes, enhanced_prompts)):
        base_progress = 0.15 + 0.75 * (i / len(keyframes))
        progress(base_progress, desc=f"🎬 Preparando entorno de generacion {i+1}/{len(keyframes)}…")
        
        mid_frame = round((kf.t_start + kf.t_end) / 2 * num_frames)
        sched.apply_for_frame(mid_frame)

        seg_frames = max(8, round((kf.t_end - kf.t_start) * num_frames))
        if any(x in model_key for x in ["CogVideoX", "Wan2.1", "Hunyuan"]):
            seg_frames = (seg_frames // 8) * 8 + 1

        try:
            # Check cache first
            cache_kwargs = {
                "model_key": model_key,
                "prompt": enhanced,
                "negative": negative,
                "num_frames": seg_frames,
                "guidance": guidance,
                "steps": steps,
                "width": width,
                "height": height,
                "seed": int(seed),
                "loras": active_lora_names or [],
                "v2v_strength": float(v2v_strength) if v2v_strength else 0.7
            }
            cached_frames = segment_cache.get_cached_segment(**cache_kwargs)
            
            if cached_frames:
                print(f"[Cache Hit] Usando segmento cacheado para keyframe {i+1}")
                progress(base_progress + 0.05, desc=f"⚡ Segmento recuperado del caché ({i+1}/{len(keyframes)})")
                frames = cached_frames
            else:
                frames = _run_pipe(
                    pipe, mode, enhanced, negative, input_image,
                    seg_frames, guidance, steps, width, height, gen,
                    progress=progress, base_prog=base_progress,
                    input_video_path=input_video_path,
                    v2v_strength=cache_kwargs["v2v_strength"],
                )
                # Save to cache asynchronously or just synchronously
                print(f"[Cache Miss] Generado y agendado para caché keyframe {i+1}")
                segment_cache.save_cached_segment(list(frames), **cache_kwargs)

            segments.append(list(frames))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error("❌ GPU sin memoria. Reduce resolución o frames.")

    cf = int(crossfade_frames) if crossfade_frames else 0
    if cf > 0 and len(segments) > 1:
        progress(0.91, desc="🔀 Aplicando interpolación crossfade entre segmentos…")
        all_frames = crossfade_segments(segments, overlap_frames=cf)
    else:
        all_frames = [f for seg in segments for f in seg]

    progress(0.92, desc="💾 Exportando MP4…")
    out_path = frames_to_mp4(all_frames, fps=fps)
    elapsed  = time.time() - t0

    deps["db"].save_generation(GenerationRecord(
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
        output_path=out_path, duration_s=elapsed, frame_count=len(all_frames),
    ))

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
