"""
modules/pipeline_utils.py
──────────────────────────
Shared pipeline management utilities used by both the Gradio UI (app.py)
and the command-line generator (generate.py).

Centralises:
  • MODELS registry            – single source of truth for all supported models
  • MODELS_CLI_ALIASES         – short CLI-friendly names → MODELS keys
  • load_pipeline()            – cached loader with GPU memory management
  • frames_to_mp4()            – frame list → MP4 export
  • extract_video_frames()     – MP4 → list of PIL frames
"""

from __future__ import annotations
import os, uuid
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace token
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# ─────────────────────────────────────────────────────────────────────────────
# Models registry — single source of truth
# ─────────────────────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
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
    "CogVideoX-5B-V2V (Vid→Vid)": {
        "repo":     "THUDM/CogVideoX-5b",
        "type":     "v2v",
        "pipeline": "CogVideoXVideoToVideoPipeline",
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
    "Wan2.1 (T2V)": {
        "repo":     "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "type":     "t2v",
        "pipeline": "WanPipeline",
    },
    "HunyuanVideo (T2V)": {
        "repo":     "tencent/HunyuanVideo",
        "type":     "t2v",
        "pipeline": "HunyuanVideoPipeline",
    },
}

# CLI-friendly short aliases → full MODELS keys
MODELS_CLI_ALIASES: dict[str, str] = {
    "cogvideox":     "CogVideoX-5B (T2V)",
    "cogvideox-i2v": "CogVideoX-5B-I2V (Img→Vid)",
    "cogvideox-v2v": "CogVideoX-5B-V2V (Vid→Vid)",
    "ltx":           "LTX-Video (T2V rápido)",
    "ltx-i2v":       "LTX-Video-I2V (Img→Vid rápido)",
    "wan":           "Wan2.1 (T2V)",
    "hunyuan":       "HunyuanVideo (T2V)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline class map (lazy import to avoid loading diffusers at module level)
# ─────────────────────────────────────────────────────────────────────────────

def _get_pipeline_cls_map() -> dict:
    """Import and return the pipeline class map. Deferred to avoid import cost."""
    from diffusers import (
        CogVideoXPipeline, CogVideoXImageToVideoPipeline,
        CogVideoXVideoToVideoPipeline,
        LTXPipeline, LTXImageToVideoPipeline,
        WanPipeline, HunyuanVideoPipeline,
    )
    return {
        "CogVideoXPipeline":              CogVideoXPipeline,
        "CogVideoXImageToVideoPipeline":  CogVideoXImageToVideoPipeline,
        "CogVideoXVideoToVideoPipeline":  CogVideoXVideoToVideoPipeline,
        "LTXPipeline":                    LTXPipeline,
        "LTXImageToVideoPipeline":        LTXImageToVideoPipeline,
        "WanPipeline":                    WanPipeline,
        "HunyuanVideoPipeline":           HunyuanVideoPipeline,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline cache (used by Gradio app — single-model at a time)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_cache: dict = {}
_current_model_key: str | None = None


def load_pipeline(
    model_key: str,
    token: str | None = None,
    progress=None,
    use_cache: bool = True,
):
    """
    Load (or return cached) a diffusers pipeline for the given model_key.

    Parameters
    ----------
    model_key : str
        A key from MODELS (e.g. "CogVideoX-5B (T2V)") or from
        MODELS_CLI_ALIASES (e.g. "cogvideox").
    token : str | None
        HuggingFace token. Falls back to HF_TOKEN env var.
    progress : callable | None
        Optional Gradio progress callback.
    use_cache : bool
        If True (default for Gradio), cache the pipeline and evict previous.
        Set to False for CLI one-shot usage.

    Returns
    -------
    pipe : DiffusionPipeline
    """
    global _pipeline_cache, _current_model_key

    # Resolve CLI alias if needed
    if model_key in MODELS_CLI_ALIASES:
        model_key = MODELS_CLI_ALIASES[model_key]

    # Cache hit
    if use_cache and model_key == _current_model_key and model_key in _pipeline_cache:
        return _pipeline_cache[model_key]

    # Evict previous model to free VRAM
    if use_cache:
        for v in list(_pipeline_cache.values()):
            del v
        torch.cuda.empty_cache()
        _pipeline_cache = {}

    cfg   = MODELS[model_key]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    cls_map = _get_pipeline_cls_map()

    msg = f"Descargando / Cargando {cfg['pipeline']} (~14GB - Tomará un momento)..."
    print(f"[INFO] {msg}")
    if progress:
        progress(0.10, desc=msg)

    hf_token = token or HF_TOKEN
    load_kwargs = {"torch_dtype": dtype}
    if hf_token:
        load_kwargs["token"] = hf_token

    pipe = cls_map[cfg["pipeline"]].from_pretrained(cfg["repo"], **load_kwargs)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    if use_cache:
        _pipeline_cache[model_key] = pipe
        _current_model_key = model_key

    return pipe


def get_model_type(model_key: str) -> str:
    """Return the type ('t2v', 'i2v', 'v2v') for a model key or alias."""
    if model_key in MODELS_CLI_ALIASES:
        model_key = MODELS_CLI_ALIASES[model_key]
    return MODELS[model_key]["type"]


# ─────────────────────────────────────────────────────────────────────────────
# Frames → MP4
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_frame(frame) -> np.ndarray:
    """
    Normalize a single frame to uint8 RGB numpy array.
    
    Handles:
      - PIL.Image → np.array
      - torch.Tensor → numpy
      - float [0,1] or [-1,1] → uint8 [0,255]
      - Already uint8 → passthrough
    """
    # Convert PIL to numpy
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"), dtype=np.uint8)

    # Convert torch tensor to numpy
    if hasattr(frame, 'cpu'):  # torch.Tensor
        frame = frame.cpu().numpy()

    arr = np.asarray(frame)

    # Handle different dtypes
    if arr.dtype in (np.float32, np.float64, np.float16):
        # Detect range
        vmin, vmax = arr.min(), arr.max()
        if vmin < -0.5:  # Likely [-1, 1] range
            arr = ((arr + 1.0) / 2.0 * 255.0)
        elif vmax <= 1.5:  # Likely [0, 1] range
            arr = (arr * 255.0)
        # else: already in ~[0, 255] range
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Ensure 3-channel RGB (handle grayscale or RGBA)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))

    return arr


def frames_to_mp4(
    frames,
    fps: int = 24,
    output_dir: str | Path = "outputs",
    output_path: str | None = None,
) -> str:
    """
    Write a list of PIL/ndarray/tensor frames to an MP4 file.
    Handles normalization from float/tensor to uint8 RGB automatically.
    """
    import imageio
    if output_path is None:
        Path(output_dir).mkdir(exist_ok=True)
        output_path = str(Path(output_dir) / f"{uuid.uuid4().hex}.mp4")

    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264",
        quality=8, pixelformat="yuv420p",
    )
    for f in frames:
        normalized = _normalize_frame(f)
        writer.append_data(normalized)
    writer.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Video file → list of PIL frames
# ─────────────────────────────────────────────────────────────────────────────

def extract_video_frames(video_path: str, max_frames: int = 49) -> list:
    """Extract frames from a video file, returning a list of PIL Images."""
    import imageio
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        frames.append(Image.fromarray(frame))
    reader.close()
    return frames
