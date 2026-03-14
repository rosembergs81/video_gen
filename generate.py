#!/usr/bin/env python3
"""
generate.py — CLI para generación headless con soporte de LoRAs y movimiento
──────────────────────────────────────────────────────────────────────────────
Ejemplos:
  # Texto → Video
  python generate.py \\
    --prompt "A dragon flying over snowy mountains" \\
    --camera "slow zoom in" \\
    --subject "subject flying through the air" \\
    --speed "slow motion" \\
    --style "cinematic, film grain" \\
    --model cogvideox \\
    --output dragon.mp4

  # Con LoRAs (uno o varios)
  python generate.py \\
    --prompt "Anime girl walking in rain" \\
    --lora "guoyww/animatediff-motion-lora-pan-left:0.8" \\
    --lora "/workspace/loras/anime_style.safetensors:0.7" \\
    --model ltx \\
    --output anime.mp4

  # Imagen → Video
  python generate.py \\
    --prompt "Ocean waves gently moving" \\
    --image my_photo.jpg \\
    --model cogvideox-i2v \\
    --output wave.mp4
"""

import argparse, sys, time, json, os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from modules.pipeline_utils import (
    MODELS_CLI_ALIASES, load_pipeline, get_model_type,
    frames_to_mp4, extract_video_frames
)

# ─────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="CLI Video Generator with LoRA + Motion support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Core
    p.add_argument("--prompt",          required=True,  help="Descripción base del video")
    p.add_argument("--negative-prompt", default="",     help="Negative prompt")
    p.add_argument("--model",           default="cogvideox",
                   choices=list(MODELS_CLI_ALIASES.keys()),     help="Modelo a usar")
    p.add_argument("--image",           default=None,   help="Imagen de entrada (I2V)")
    p.add_argument("--video",           default=None,   help="Video de entrada (V2V)")
    p.add_argument("--v2v-strength",    type=float, default=0.5,
                   help="Fuerza de transformación V2V (0.1-1.0)")
    p.add_argument("--output",          default="output.mp4")

    # Motion tags (strings que se añaden al prompt)
    motion = p.add_argument_group("Motion")
    motion.add_argument("--camera",  default="",
                        help="Tag de movimiento de cámara, ej: 'slow zoom in'")
    motion.add_argument("--subject", default="",
                        help="Tag de movimiento del sujeto, ej: 'subject running'")
    motion.add_argument("--speed",   default="",
                        help="Tag de velocidad, ej: 'slow motion'")
    motion.add_argument("--style",   default="",
                        help="Tag de estilo, ej: 'cinematic, film grain'")

    # LoRAs: formato "source:scale" o solo "source" (scale default 0.8)
    # Puede repetirse: --lora a --lora b
    p.add_argument("--lora", action="append", default=[], metavar="SOURCE[:SCALE]",
                   help="LoRA HF repo o path local, opcionalmente con escala ej: repo:0.7")
    p.add_argument("--lora-weight-name", action="append", default=[],
                   metavar="FILENAME",
                   help="weight_name para cada LoRA en el mismo orden (opcional)")

    # Generation params
    gen = p.add_argument_group("Generation")
    gen.add_argument("--frames",   type=int,   default=49)
    gen.add_argument("--fps",      type=int,   default=24)
    gen.add_argument("--steps",    type=int,   default=50)
    gen.add_argument("--guidance", type=float, default=6.0)
    gen.add_argument("--width",    type=int,   default=720)
    gen.add_argument("--height",   type=int,   default=480)
    gen.add_argument("--seed",     type=int,   default=-1)
    gen.add_argument("--bf16",     action="store_true", default=True,
                     help="Usar bfloat16 si está disponible (default: True)")
    # HuggingFace token (alternativa a export HF_TOKEN=...)
    p.add_argument("--hf-token", default=None,
                   help="Token de HuggingFace (alternativa a variable de entorno HF_TOKEN)")
    return p.parse_args()

# ─────────────────────────────────────────────
# LoRA loader
# ─────────────────────────────────────────────
def load_loras(pipe, lora_args: list, weight_names: list, token: str | None = None):
    """
    lora_args   : list of "source" or "source:scale"
    weight_names: list of optional filenames (same order, or empty)
    """
    if not lora_args:
        return

    adapters, weights = [], []
    for i, raw in enumerate(lora_args):
        parts  = raw.rsplit(":", 1)
        source = parts[0]
        scale  = float(parts[1]) if len(parts) == 2 else 0.8
        wname  = weight_names[i] if i < len(weight_names) else None
        adapter_id = f"lora_{i}"

        print(f"[LoRA] Cargando '{source}' (escala={scale}) …")
        try:
            kwargs = {"adapter_name": adapter_id}
            if wname:
                kwargs["weight_name"] = wname
            if token:
                kwargs["token"] = token
            pipe.load_lora_weights(source, **kwargs)
            adapters.append(adapter_id)
            weights.append(scale)
        except Exception as e:
            print(f"[LoRA] ⚠️  Error: {e}", file=sys.stderr)

    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)
        print(f"[LoRA] Activas: {adapters} con escalas {weights}")

# ─────────────────────────────────────────────
# Prompt assembly
# ─────────────────────────────────────────────
def build_prompt(base, camera, subject, speed, style):
    extras = [t for t in [camera, subject, speed, style] if t.strip()]
    if not extras:
        return base
    return base.rstrip(" ,.") + ", " + ", ".join(extras)


# Main
# ─────────────────────────────────────────────
def main():
    args  = parse_args()
    dtype = (torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported()
             else torch.float16)

    # Resolve HF token: CLI arg takes priority over env var
    hf_token = (args.hf_token
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN"))
    if hf_token:
        print(f"[INFO] Token HuggingFace activo (longitud={len(hf_token)})")
    else:
        print("[WARN] Sin HF_TOKEN. Modelos con gating podrían fallar. "
              "Usa --hf-token o: export HF_TOKEN=hf_...")

    # Build final prompt
    final_prompt = build_prompt(
        args.prompt, args.camera, args.subject, args.speed, args.style
    )
    print(f"\n[Prompt] {final_prompt}\n")

    # Load model
    pipe = load_pipeline(args.model, token=hf_token, use_cache=False)
    mode = get_model_type(args.model)

    # Load LoRAs
    load_loras(pipe, args.lora, args.lora_weight_name, token=hf_token)

    # Generator
    gen = (torch.Generator("cuda").manual_seed(args.seed)
           if args.seed != -1 else None)

    print(f"[INFO] Generando {args.frames} frames ({mode}) …")
    t0 = time.time()

    if mode == "t2v":
        result = pipe(
            prompt=final_prompt,
            negative_prompt=args.negative_prompt or None,
            num_frames=args.frames,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            width=args.width,
            height=args.height,
            generator=gen,
        )
    elif mode == "v2v":
        if not args.video:
            sys.exit("❌ --video es requerido para modelos V2V")
        video_frames = extract_video_frames(args.video, max_frames=args.frames)
        # Resize frames to 720x480 (CogVideoX requirement)
        video_frames = [f.resize((720, 480), Image.LANCZOS) for f in video_frames]
        result = pipe(
            video=video_frames,
            prompt=final_prompt,
            negative_prompt=args.negative_prompt or None,
            num_frames=args.frames,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            strength=args.v2v_strength,
            generator=gen,
        )
    else:
        if not args.image:
            sys.exit("❌ --image es requerido para modelos I2V")
        img  = Image.open(args.image).convert("RGB")
        # Resize to 720x480 (CogVideoX I2V requirement)
        img  = img.resize((720, 480), Image.LANCZOS)
        result = pipe(
            image=img,
            prompt=final_prompt,
            negative_prompt=args.negative_prompt or None,
            num_frames=args.frames,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=gen,
        )

    frames_to_mp4(result.frames[0], fps=args.fps, output_path=args.output)
    print(f"\n✅ Video guardado en '{args.output}'  ({time.time()-t0:.1f}s)\n")

if __name__ == "__main__":
    main()
