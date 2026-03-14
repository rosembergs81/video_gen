"""
modules/story_mode.py
───────────────────────
Lógica para generación secuencial de videos largos con continuidad visual.
"""
import time
import re
import torch
import gradio as gr
from PIL import Image

from modules.database import GenerationRecord
from modules.pipeline_utils import load_pipeline, frames_to_mp4
from modules.generation import crossfade_segments


def _get_i2v_pipe(progress=None):
    """Load the I2V pipeline for story continuity chaining."""
    return load_pipeline("CogVideoX-5B-I2V (Img→Vid)", progress=progress)


def generate_story(
    story_script, negative_preset,
    lighting, cinematography, quality, atmosphere, genre,
    frames_per_scene, fps, guidance, steps, seed,
    crossfade_frames, reference_image,
    # Injected singletons:
    deps: dict,
    progress=gr.Progress(track_tqdm=False),
):
    if not story_script or not story_script.strip():
        raise gr.Error("⚠️ El guión no puede estar vacío.")

    scenes = []
    for line in story_script.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^\d+[\.\:\)\-]\s*", "", line)
        if line:
            scenes.append(line)

    if not scenes:
        raise gr.Error("⚠️ No se encontraron escenas en el guión.")
    if len(scenes) > 20:
        raise gr.Error("⚠️ Máximo 20 escenas por historia.")

    negative = deps["enhancer"].get_negative(negative_preset)
    seg_frames = int(frames_per_scene)
    seg_frames = (seg_frames // 8) * 8 + 1

    gen = (torch.Generator("cuda").manual_seed(int(seed)) if int(seed) != -1 else None)
    t0 = time.time()
    all_segments = []
    last_frame_pil = None
    scene_log = []

    if reference_image is not None:
        last_frame_pil = Image.fromarray(reference_image).convert("RGB")

    for i, scene_prompt in enumerate(scenes):
        prog_base = i / len(scenes)
        progress(prog_base, desc=f"🎬 Escena {i+1}/{len(scenes)}: Preparando...")

        enhanced = deps["enhancer"].enhance(
            base=scene_prompt,
            lighting=None if lighting == "—" else lighting,
            cinematography=None if cinematography == "—" else cinematography,
            quality=None if quality == "—" else quality,
            atmosphere=None if atmosphere == "—" else atmosphere,
            genre=None if genre == "—" else genre,
        )
        words = enhanced.split()
        if len(words) > 200:
            enhanced = " ".join(words[:200])

        scene_log.append(f"`{i+1}.` {enhanced[:100]}{'…' if len(enhanced) > 100 else ''}")

        def _make_step_cb(scene_idx, pb):
            def step_cb(pipe_cls, step, timestep, cb_kwargs):
                progress(pb, desc=f"⏳ Escena {scene_idx+1}: Paso {step+1}/{steps}...")
                return cb_kwargs
            return step_cb

        try:
            if i == 0 and last_frame_pil is None:
                progress(prog_base + 0.02, desc=f"🎬 Escena 1/{len(scenes)}: Generando con T2V...")
                pipe_t2v = load_pipeline("CogVideoX-5B (T2V)", progress=progress)

                frames = pipe_t2v(
                    prompt=enhanced, negative_prompt=negative or None,
                    num_frames=seg_frames, guidance_scale=guidance,
                    num_inference_steps=steps, generator=gen,
                    callback_on_step_end=_make_step_cb(i, prog_base),
                ).frames[0]

            else:
                progress(prog_base + 0.02,
                         desc=f"🎬 Escena {i+1}/{len(scenes)}: Generando con I2V (continuidad)...")

                pipe_i2v = _get_i2v_pipe(progress=progress)
                seed_img = last_frame_pil.resize((720, 480), Image.LANCZOS)

                frames = pipe_i2v(
                    image=seed_img,
                    prompt=enhanced, negative_prompt=negative or None,
                    num_frames=seg_frames, guidance_scale=guidance,
                    num_inference_steps=steps, generator=gen,
                    callback_on_step_end=_make_step_cb(i, prog_base),
                ).frames[0]

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error(
                f"❌ GPU sin memoria en escena {i+1}/{len(scenes)}. "
                f"Reduce frames por escena o el número de escenas."
            )

        frame_list = list(frames)
        all_segments.append(frame_list)

        last_raw = frame_list[-1]
        if isinstance(last_raw, Image.Image):
            last_frame_pil = last_raw
        else:
            last_frame_pil = Image.fromarray(last_raw)

    cf = int(crossfade_frames) if crossfade_frames else 4
    if cf > 0 and len(all_segments) > 1:
        progress(0.92, desc="🔀 Aplicando crossfade entre escenas...")
        final_frames = crossfade_segments(all_segments, overlap_frames=cf)
    else:
        final_frames = [f for seg in all_segments for f in seg]

    progress(0.95, desc="💾 Exportando historia completa...")
    out_path = frames_to_mp4(final_frames, fps=int(fps))
    elapsed = time.time() - t0

    deps["db"].save_generation(GenerationRecord(
        model="Story Mode (T2V→I2V chain)",
        prompt=story_script[:500],
        negative=negative,
        params=dict(
            frames_per_scene=int(frames_per_scene), fps=int(fps),
            guidance=guidance, steps=steps,
            seed=int(seed), total_scenes=len(scenes),
            crossfade=cf,
        ),
        loras=[],
        motion_tags=dict(
            lighting=lighting, cinematography=cinematography,
            quality=quality, atmosphere=atmosphere, genre=genre,
        ),
        output_path=out_path,
        duration_s=elapsed,
        frame_count=len(final_frames),
    ))

    total_secs = len(final_frames) / max(1, int(fps))
    info = (
        f"✅ **Historia generada:** {len(scenes)} escenas · "
        f"**{len(final_frames)} frames** · **{total_secs:.1f}s** de video · "
        f"generado en **{elapsed:.1f}s**\n\n"
        f"**Escenas procesadas:**\n" +
        "\n".join(scene_log) +
        f"\n\n📁 `{out_path}`"
    )
    return out_path, info
