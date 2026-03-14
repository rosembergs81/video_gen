"""
modules/lora_scheduler.py
──────────────────────────
Permite que los pesos de las LoRAs cambien dinámicamente por escena o frame.
Ejemplo: una LoRA de "pan_left" activa solo durante el movimiento de cámara,
         una LoRA de "anime" con fade-in en la segunda mitad del video.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoRAEntry:
    """A single LoRA with its source, range, and weight curve."""
    lora_id:     str              # internal adapter name
    source:      str              # HF repo or local path
    weight_name: str | None = None
    # Active range (frame-based)
    frame_start: int   = 0
    frame_end:   int   = -1       # -1 = until last frame
    # Weight settings
    base_scale:  float = 0.8
    # Optional curve: callable(t 0..1) → scale multiplier
    curve: Callable[[float], float] | None = None
    category: str = "estilo"      # for display purposes
    label: str = ""

    def scale_at(self, frame: int, total_frames: int) -> float:
        """Returns the effective LoRA scale at this frame."""
        end = self.frame_end if self.frame_end != -1 else total_frames
        if not (self.frame_start <= frame <= end):
            return 0.0                         # outside active range → inactive

        if self.curve is None:
            return self.base_scale

        t = (frame - self.frame_start) / max(end - self.frame_start, 1)
        return self.base_scale * self.curve(t)

    def is_active(self, frame: int, total_frames: int) -> bool:
        return self.scale_at(frame, total_frames) > 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Built-in curves
# ─────────────────────────────────────────────────────────────────────────────

def curve_constant(t: float) -> float:
    return 1.0

def curve_fade_in(t: float) -> float:
    return t * t * (3 - 2 * t)              # smoothstep

def curve_fade_out(t: float) -> float:
    return 1.0 - curve_fade_in(t)

def curve_fade_in_out(t: float) -> float:
    if t < 0.5:
        return curve_fade_in(t * 2) * 0.5
    return (1.0 - curve_fade_in((t - 0.5) * 2)) * 0.5 + 0.5

def curve_pulse(t: float) -> float:
    """Peak in the middle, 0 at edges."""
    import math
    return math.sin(math.pi * t)

CURVES = {
    "constant":    curve_constant,
    "fade_in":     curve_fade_in,
    "fade_out":    curve_fade_out,
    "fade_in_out": curve_fade_in_out,
    "pulse":       curve_pulse,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic LoRA Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class DynamicLoRASchedule:
    """
    Manages a collection of LoRAs with per-frame weight scheduling.

    Usage:
        sched = DynamicLoRASchedule(pipeline, total_frames=49)
        sched.add("zoom_lora", "guoyww/animatediff-motion-lora-zoom-in",
                  frame_start=0, frame_end=20, base_scale=0.9, curve="fade_out")
        sched.add("anime_lora", "/loras/anime.safetensors",
                  frame_start=24, frame_end=48, base_scale=0.8, curve="fade_in")
        # At generation time:
        adapters, weights = sched.get_active(frame=10)
    """

    def __init__(self, pipeline=None, total_frames: int = 49):
        self.pipeline      = pipeline
        self.total_frames  = total_frames
        self.loras: list[LoRAEntry] = []
        self._loaded: set[str] = set()

    def add(
        self,
        lora_id:     str,
        source:      str,
        frame_start: int   = 0,
        frame_end:   int   = -1,
        base_scale:  float = 0.8,
        curve:       str   = "constant",
        weight_name: str | None = None,
        category:    str   = "estilo",
        label:       str   = "",
    ):
        entry = LoRAEntry(
            lora_id=lora_id,
            source=source,
            weight_name=weight_name,
            frame_start=frame_start,
            frame_end=frame_end if frame_end != -1 else self.total_frames,
            base_scale=base_scale,
            curve=CURVES.get(curve, curve_constant),
            category=category,
            label=label or lora_id,
        )
        self.loras.append(entry)

    def add_from_dict(self, entries: list[dict]):
        """Bulk-add from list of dicts (e.g. from loras/index.json)."""
        for e in entries:
            self.add(
                lora_id=e["name"].replace(" ", "_").lower(),
                source=e["source"],
                base_scale=float(e.get("scale", 0.8)),
                weight_name=e.get("weight_name"),
                category=e.get("category", "estilo"),
                label=e.get("name", ""),
                frame_start=int(e.get("frame_start", 0)),
                frame_end=int(e.get("frame_end", -1)),
                curve=e.get("curve", "constant"),
            )

    # ── Per-frame query ──────────────────────────────────────────────────────

    def get_active(
        self, frame: int
    ) -> tuple[list[str], list[float]]:
        """
        Returns (adapter_names, weights) for all active LoRAs at this frame.
        Ready to pass directly to pipeline.set_adapters().
        """
        adapters, weights = [], []
        for lora in self.loras:
            scale = lora.scale_at(frame, self.total_frames)
            if scale > 1e-4:
                adapters.append(lora.lora_id)
                weights.append(round(scale, 3))
        return adapters, weights

    def preload_all(self):
        """
        Load all LoRAs upfront into VRAM/GPU via the pipeline to avoid pauses 
        and memory fragmentation during the segmented generation loop.
        """
        if self.pipeline is None:
            return
            
        for lora in self.loras:
            if lora.lora_id not in self._loaded:
                try:
                    kwargs = {"adapter_name": lora.lora_id}
                    if lora.weight_name:
                        kwargs["weight_name"] = lora.weight_name
                    self.pipeline.load_lora_weights(lora.source, **kwargs)
                    self._loaded.add(lora.lora_id)
                    print(f"[LoRAScheduler] Pre-loaded '{lora.lora_id}'")
                except Exception as e:
                    print(f"[LoRAScheduler] ⚠️ Failed to pre-load '{lora.lora_id}': {e}")

    def apply_for_frame(self, frame: int):
        """
        Set LoRA weights for the given frame on the attached pipeline.
        Should be called before generating each keyframe batch.
        """
        if self.pipeline is None:
            return

        adapters, weights = self.get_active(frame)

        # Fallback to load if not preloaded (should be rare)
        for adapter in adapters:
            if adapter not in self._loaded:
                lora = next(l for l in self.loras if l.lora_id == adapter)
                try:
                    kwargs = {"adapter_name": lora.lora_id}
                    if lora.weight_name:
                        kwargs["weight_name"] = lora.weight_name
                    self.pipeline.load_lora_weights(lora.source, **kwargs)
                    self._loaded.add(lora.lora_id)
                    print(f"[LoRAScheduler] Lazy-loaded '{lora.lora_id}'")
                except Exception as e:
                    print(f"[LoRAScheduler] ⚠️ Failed to lazy-load '{lora.lora_id}': {e}")

        if adapters:
            self.pipeline.set_adapters(adapters, adapter_weights=weights)
            print(f"[LoRAScheduler] frame {frame}: {list(zip(adapters, weights))}")
        else:
            # No active LoRAs — disable all
            if hasattr(self.pipeline, "disable_lora"):
                self.pipeline.disable_lora()
            else:
                self.pipeline.set_adapters([], adapter_weights=[])

    # ── Visualization ────────────────────────────────────────────────────────

    def schedule_summary(self) -> str:
        """Human-readable schedule overview."""
        lines = [f"LoRA Schedule ({len(self.loras)} entries, {self.total_frames} frames)"]
        for lora in self.loras:
            end = lora.frame_end if lora.frame_end != -1 else self.total_frames
            bar_len = 30
            bar = [" "] * bar_len
            for i in range(bar_len):
                f = int(i / bar_len * self.total_frames)
                if lora.scale_at(f, self.total_frames) > 1e-4:
                    bar[i] = "█"
            lines.append(
                f"  {lora.label:20s} [{lora.frame_start:3d}–{end:3d}] "
                f"scale={lora.base_scale:.2f}  |{''.join(bar)}|"
            )
        return "\n".join(lines)

    def to_ui_rows(self) -> list[dict]:
        """For table display in Gradio."""
        rows = []
        for lora in self.loras:
            end = lora.frame_end if lora.frame_end != -1 else self.total_frames
            rows.append({
                "Nombre":      lora.label,
                "Fuente":      lora.source,
                "Frames":      f"{lora.frame_start} → {end}",
                "Escala base": lora.base_scale,
                "Curva":       next(
                    (k for k, v in CURVES.items() if v is lora.curve),
                    "custom"
                ),
                "Categoría":   lora.category,
            })
        return rows
