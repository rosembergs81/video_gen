"""
modules/scene_graph.py
──────────────────────
Mantiene relaciones espaciales y de interacción entre múltiples sujetos
a lo largo del tiempo. Permite rastrear oclusión, contacto y coherencia
de posiciones entre frames.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Spatial relationships vocabulary
# ─────────────────────────────────────────────────────────────────────────────

SPATIAL_RELATIONS = {
    "on_top_of":    "resting on top of",
    "beside":       "standing beside",
    "behind":       "positioned behind",
    "in_front_of":  "in front of",
    "approaching":  "moving towards",
    "leaving":      "moving away from",
    "touching":     "in contact with",
    "holding":      "holding",
    "surrounding":  "surrounding",
    "inside":       "inside",
    "overlapping":  "partially overlapping",
}

# For temporal coherence checks
INCOMPATIBLE_PAIRS = {
    # (action_a, action_b) — b cannot follow a in <1 second
    ("jumping", "sitting"),
    ("running", "sleeping"),
    ("exploding", "standing_still"),
    ("flying_fast", "gentle_landing"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Object3D:
    obj_id: str
    label: str
    category: str = "personaje"         # personaje | animal | objeto
    position: tuple = (0.5, 0.5, 0.0)  # (x, y, z) normalized 0..1
    size: float = 1.0                   # relative size
    current_action: str = ""
    visible: bool = True
    occlusion_by: Optional[str] = None  # obj_id that hides this object


@dataclass
class Interaction:
    subject_id: str
    object_id:  str
    relation:   str         # key from SPATIAL_RELATIONS
    frame_start: int = 0
    frame_end:   int = -1   # -1 = until end
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Scene Graph
# ─────────────────────────────────────────────────────────────────────────────

class SceneGraph:
    """
    Usage:
        graph = SceneGraph()
        graph.add_object("CHAR_1", "Lion", "animal", position=(0.3, 0.5, 0))
        graph.add_object("CHAR_2", "Bird", "animal", position=(0.9, 0.3, 0))
        graph.add_interaction("CHAR_2", "CHAR_1", "on_top_of", frame_start=30)
        prompt_suffix = graph.to_prompt_suffix(frame=35)
    """

    def __init__(self):
        self.objects: dict[str, Object3D]    = {}
        self.interactions: list[Interaction] = []
        self._frame_history: list[dict]      = []   # snapshots for tracking

    # ── Object management ────────────────────────────────────────────────────

    def add_object(
        self,
        obj_id: str,
        label: str,
        category: str = "personaje",
        position: tuple = (0.5, 0.5, 0.0),
        size: float = 1.0,
    ) -> Object3D:
        obj = Object3D(
            obj_id=obj_id, label=label,
            category=category, position=position, size=size,
        )
        self.objects[obj_id] = obj
        return obj

    def update_position(self, obj_id: str, position: tuple, frame: int = 0):
        if obj_id in self.objects:
            self.objects[obj_id].position = position

    def set_action(self, obj_id: str, action: str):
        if obj_id in self.objects:
            self.objects[obj_id].current_action = action

    # ── Interaction management ───────────────────────────────────────────────

    def add_interaction(
        self,
        subject_id: str,
        object_id: str,
        relation: str,
        frame_start: int = 0,
        frame_end: int = -1,
        description: str = "",
    ) -> Interaction:
        rel_desc = SPATIAL_RELATIONS.get(relation, relation)
        inter = Interaction(
            subject_id=subject_id,
            object_id=object_id,
            relation=relation,
            frame_start=frame_start,
            frame_end=frame_end,
            description=description or rel_desc,
        )
        self.interactions.append(inter)
        # Handle occlusion
        if relation in ("on_top_of", "inside", "overlapping"):
            if object_id in self.objects:
                self.objects[object_id].occlusion_by = subject_id
        return inter

    # ── Spatial utilities ────────────────────────────────────────────────────

    def distance(self, id_a: str, id_b: str) -> float:
        """Euclidean distance between two objects."""
        if id_a not in self.objects or id_b not in self.objects:
            return float("inf")
        a = self.objects[id_a].position
        b = self.objects[id_b].position
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    def objects_in_contact(self, threshold: float = 0.15) -> list[tuple]:
        """Returns pairs of objects that are spatially close."""
        ids = list(self.objects.keys())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if self.distance(ids[i], ids[j]) < threshold:
                    pairs.append((ids[i], ids[j]))
        return pairs

    def get_active_interactions(self, frame: int) -> list[Interaction]:
        active = []
        for inter in self.interactions:
            start_ok = frame >= inter.frame_start
            end_ok   = inter.frame_end == -1 or frame <= inter.frame_end
            if start_ok and end_ok:
                active.append(inter)
        return active

    # ── Prompt generation ────────────────────────────────────────────────────

    def to_prompt_suffix(self, frame: int = 0) -> str:
        """
        Generates a prompt suffix describing spatial relationships at a given frame.
        Example: "bird resting on top of lion's head, lion looking upward"
        """
        parts = []
        active = self.get_active_interactions(frame)

        for inter in active:
            subj = self.objects.get(inter.subject_id)
            obj  = self.objects.get(inter.object_id)
            if subj and obj:
                rel  = SPATIAL_RELATIONS.get(inter.relation, inter.relation)
                parts.append(f"{subj.label} {rel} {obj.label}")

        for obj in self.objects.values():
            if obj.current_action:
                parts.append(f"{obj.label} {obj.current_action}")

        return ", ".join(parts)

    def snapshot(self, frame: int):
        """Save current state for later analysis."""
        snap = {
            "frame": frame,
            "objects": {
                oid: {
                    "position": obj.position,
                    "action":   obj.current_action,
                    "visible":  obj.visible,
                }
                for oid, obj in self.objects.items()
            },
        }
        self._frame_history.append(snap)

    # ── Serialization ────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"SceneGraph: {len(self.objects)} objects, "
                 f"{len(self.interactions)} interactions"]
        for oid, obj in self.objects.items():
            lines.append(f"  {oid}: {obj.label} @ {obj.position} — {obj.current_action}")
        for inter in self.interactions:
            lines.append(f"  {inter.subject_id} —[{inter.relation}]→ {inter.object_id} "
                         f"frames {inter.frame_start}–{inter.frame_end}")
        return "\n".join(lines)
