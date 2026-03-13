"""
modules/scene_parser.py
───────────────────────
Convierte un prompt narrativo en una secuencia de keyframes con acciones
encadenadas. Soporta dos modos:

  1. Sintaxis estructurada  →  bloques [SCENE], [CHAR_x], [INTERACTION]
  2. Lenguaje natural       →  detecta verbos de transición y los separa

Salida estándar:
    List[SceneKeyframe]  — ordenados por t_start (0.0..1.0)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CharacterState:
    char_id: str                            # "CHAR_1", "CHAR_2" …
    label: str                              # "Lion", "Bird" …
    mood: str = ""                          # "aggressive", "calm" …
    body_parts: dict = field(default_factory=dict)  # {"head": "turning left"}
    position: Optional[str] = None          # "left frame", "center" …
    action: str = ""                        # high-level action string


@dataclass
class SceneKeyframe:
    index: int                              # keyframe order (0-based)
    t_start: float                          # normalized time 0.0 – 1.0
    t_end: float
    description: str                        # full assembled prompt for this segment
    scene_context: str = ""                 # environment / lighting
    characters: list[CharacterState] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)
    camera_hint: str = ""                   # embedded camera instruction


# ─────────────────────────────────────────────────────────────────────────────
# Transition markers — words that split a prompt into sequential actions
# ─────────────────────────────────────────────────────────────────────────────

_TRANSITION_WORDS = [
    r"\bthen\b", r"\bafter\b", r"\bafterwards\b", r"\bnext\b",
    r"\bfinally\b", r"\bsuddenly\b", r"\bslowly\b(?=\s+\w+s\b)",
    r"\buntil\b", r"\bwhen\b", r"\bas\s+it\b",
    # Spanish
    r"\bluego\b", r"\bdespués\b", r"\bcuando\b", r"\bfinalmente\b",
    r"\bde\s+repente\b", r"\ba\s+continuación\b", r"\bmientras\b",
    r"\b→\b", r"\b->\b",
]

_TRANSITION_RE = re.compile(
    "|".join(_TRANSITION_WORDS), re.IGNORECASE
)

# Structured block patterns
_SCENE_RE       = re.compile(r"\[SCENE:\s*(.+?)\]", re.IGNORECASE)
_CHAR_RE        = re.compile(r"\[CHAR_(\d+):\s*(.+?)\]", re.IGNORECASE)
_BODYPART_RE    = re.compile(r"-\s*CHAR_(\d+)\.(\w+):\s*(.+)")
_INTERACTION_RE = re.compile(r"\[INTERACTION:\s*(.+?)\]", re.IGNORECASE)
_CAM_RE         = re.compile(
    r"\[(CAM|CAMERA):\s*(.+?)\]", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class SceneParser:
    """
    Usage:
        parser = SceneParser()
        keyframes = parser.parse(prompt_text, total_frames=49)
    """

    def parse(self, text: str, total_frames: int = 49) -> list[SceneKeyframe]:
        text = text.strip()
        if self._is_structured(text):
            return self._parse_structured(text, total_frames)
        return self._parse_natural(text, total_frames)

    # ── detection ────────────────────────────────────────────────────────────

    def _is_structured(self, text: str) -> bool:
        return bool(_SCENE_RE.search(text) or _CHAR_RE.search(text))

    # ── natural language → keyframes ─────────────────────────────────────────

    def _parse_natural(self, text: str, total_frames: int) -> list[SceneKeyframe]:
        segments = _TRANSITION_RE.split(text)
        segments = [s.strip() for s in segments if s.strip()]

        if len(segments) == 1:
            # Single-shot prompt: one keyframe covering the whole video
            return [SceneKeyframe(
                index=0, t_start=0.0, t_end=1.0,
                description=text,
            )]

        keyframes = []
        n = len(segments)
        for i, seg in enumerate(segments):
            t_start = i / n
            t_end   = (i + 1) / n
            kf = SceneKeyframe(
                index=i,
                t_start=round(t_start, 3),
                t_end=round(t_end, 3),
                description=seg,
            )
            # Inherit scene context from first segment
            if i == 0:
                kf.scene_context = seg
            keyframes.append(kf)

        return keyframes

    # ── structured → keyframes ───────────────────────────────────────────────

    def _parse_structured(self, text: str, total_frames: int) -> list[SceneKeyframe]:
        lines = text.split("\n")

        scene_ctx   = ""
        characters  = {}     # id → CharacterState
        interactions = []
        camera_hint = ""
        current_char_id = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # [SCENE: ...]
            m = _SCENE_RE.match(line)
            if m:
                scene_ctx = m.group(1).strip()
                continue

            # [CHAR_N: Label, mood]
            m = _CHAR_RE.match(line)
            if m:
                cid   = f"CHAR_{m.group(1)}"
                parts = [p.strip() for p in m.group(2).split(",")]
                label = parts[0] if parts else cid
                mood  = parts[1] if len(parts) > 1 else ""
                characters[cid] = CharacterState(
                    char_id=cid, label=label, mood=mood
                )
                current_char_id = cid
                continue

            # - CHAR_N.bodypart: description
            m = _BODYPART_RE.match(line)
            if m:
                cid   = f"CHAR_{m.group(1)}"
                part  = m.group(2)
                desc  = m.group(3).strip()
                if cid in characters:
                    characters[cid].body_parts[part] = desc
                continue

            # [INTERACTION: ...]
            m = _INTERACTION_RE.match(line)
            if m:
                interactions.append(m.group(1).strip())
                continue

            # [CAM: ...]
            m = _CAM_RE.match(line)
            if m:
                camera_hint = m.group(2).strip()
                continue

        # Build a single rich keyframe from structured input
        description = self._assemble_description(
            scene_ctx, list(characters.values()), interactions
        )

        kf = SceneKeyframe(
            index=0, t_start=0.0, t_end=1.0,
            description=description,
            scene_context=scene_ctx,
            characters=list(characters.values()),
            interactions=interactions,
            camera_hint=camera_hint,
        )
        return [kf]

    # ── assembler ────────────────────────────────────────────────────────────

    def _assemble_description(
        self,
        scene: str,
        chars: list[CharacterState],
        interactions: list[str],
    ) -> str:
        parts = [scene] if scene else []

        for c in chars:
            char_desc = f"{c.label}"
            if c.mood:
                char_desc += f" ({c.mood})"
            if c.action:
                char_desc += f" {c.action}"
            bp_descs = [f"{p} {v}" for p, v in c.body_parts.items()]
            if bp_descs:
                char_desc += " with " + ", ".join(bp_descs)
            parts.append(char_desc)

        parts.extend(interactions)
        return ", ".join(parts)

    # ── helpers ──────────────────────────────────────────────────────────────

    def distribute_frames(
        self,
        keyframes: list[SceneKeyframe],
        total_frames: int,
    ) -> list[tuple[SceneKeyframe, int, int]]:
        """
        Returns [(keyframe, frame_start, frame_end), …]
        """
        result = []
        for kf in keyframes:
            f_start = round(kf.t_start * total_frames)
            f_end   = round(kf.t_end   * total_frames)
            result.append((kf, f_start, f_end))
        return result

    def to_prompt_list(self, keyframes: list[SceneKeyframe]) -> list[str]:
        """Returns just the description strings, for sequential generation."""
        return [kf.description for kf in keyframes]


# ─────────────────────────────────────────────────────────────────────────────
# Body parts registry
# ─────────────────────────────────────────────────────────────────────────────

BODY_PARTS = {
    "personaje": ["cabeza", "torso", "brazos", "manos", "piernas", "ojos", "boca"],
    "animal":    ["cabeza", "cuerpo", "patas", "cola", "alas", "boca", "orejas"],
    "objeto":    ["parte_principal", "detalles", "accesorios", "superficie"],
}
