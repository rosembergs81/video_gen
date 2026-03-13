"""
modules/lora_recommender.py
────────────────────────────
Sistema de recomendación automática de LoRAs basado en:
  • Acciones detectadas (ActionExtractor output)
  • Estilo visual del prompt
  • Modelo seleccionado
  • Historial de LoRAs exitosas (base de datos)

Scoring:
  • Relevance  — ¿cuánto coincide con las acciones detectadas?
  • Popularity — ¿cuántas veces se usó con buen resultado en el historial?
  • Compatibility — ¿es compatible con el modelo base?

Salida: List[LoRARecommendation]  ordenada por score descendente
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Built-in LoRA knowledge base
# (fuente, categoría, compatibilidad de modelo, trigger words, action tags)
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_LORA_DB: list[dict] = [
    # ── Camera motion LoRAs ──────────────────────────────────────────────────
    {
        "name":         "Zoom In",
        "source":       "guoyww/animatediff-motion-lora-zoom-in",
        "category":     "cámara",
        "models":       ["LTX-Video", "AnimateDiff"],
        "action_tags":  ["camera_zoom", "zoom_in", "approaching", "dolly_forward"],
        "style_tags":   ["zoom", "dramatic", "focus"],
        "trigger":      "",
        "recommended_scale": 0.85,
        "description":  "Zoom in suave sobre el sujeto principal",
        "use_with":     ["slow_motion", "dramatic", "focus"],
    },
    {
        "name":         "Zoom Out",
        "source":       "guoyww/animatediff-motion-lora-zoom-out",
        "category":     "cámara",
        "models":       ["LTX-Video", "AnimateDiff"],
        "action_tags":  ["camera_zoom", "zoom_out", "retreating", "reveal"],
        "style_tags":   ["reveal", "epic", "wide"],
        "trigger":      "",
        "recommended_scale": 0.85,
        "description":  "Zoom out que revela el entorno",
        "use_with":     ["epic", "aerial", "reveal"],
    },
    {
        "name":         "Pan Left",
        "source":       "guoyww/animatediff-motion-lora-pan-left",
        "category":     "cámara",
        "models":       ["LTX-Video", "AnimateDiff"],
        "action_tags":  ["camera_pan", "looking", "turning", "moving_left"],
        "style_tags":   ["tracking", "follow", "landscape"],
        "trigger":      "",
        "recommended_scale": 0.8,
        "description":  "Paneo de cámara hacia la izquierda",
        "use_with":     ["landscape", "tracking", "running"],
    },
    {
        "name":         "Pan Right",
        "source":       "guoyww/animatediff-motion-lora-pan-right",
        "category":     "cámara",
        "models":       ["LTX-Video", "AnimateDiff"],
        "action_tags":  ["camera_pan", "looking", "turning", "moving_right"],
        "style_tags":   ["tracking", "follow", "landscape"],
        "trigger":      "",
        "recommended_scale": 0.8,
        "description":  "Paneo de cámara hacia la derecha",
        "use_with":     ["landscape", "tracking", "running"],
    },
    {
        "name":         "Tilt Up",
        "source":       "guoyww/animatediff-motion-lora-tilt-up",
        "category":     "cámara",
        "models":       ["LTX-Video", "AnimateDiff"],
        "action_tags":  ["camera_tilt", "looking_up", "flying", "jumping", "rising"],
        "style_tags":   ["epic", "reveal", "sky"],
        "trigger":      "",
        "recommended_scale": 0.8,
        "description":  "Inclinación de cámara hacia arriba",
        "use_with":     ["flying", "jumping", "dramatic"],
    },
    {
        "name":         "Rolling Clockwise",
        "source":       "guoyww/animatediff-motion-lora-rolling-clockwise",
        "category":     "movimiento",
        "models":       ["LTX-Video", "AnimateDiff"],
        "action_tags":  ["spinning", "rotating", "rolling", "camera_orbit"],
        "style_tags":   ["dynamic", "action", "disorienting"],
        "trigger":      "",
        "recommended_scale": 0.7,
        "description":  "Rotación de cámara en sentido horario",
        "use_with":     ["action", "spinning", "dramatic"],
    },
    # ── CogVideoX specific ───────────────────────────────────────────────────
    {
        "name":         "CogVideoX Movie Style",
        "source":       "TheDenouement/cogvideox-movie-lora",
        "category":     "estilo",
        "models":       ["CogVideoX"],
        "action_tags":  [],
        "style_tags":   ["cinematic", "film", "movie", "dramatic"],
        "trigger":      "",
        "recommended_scale": 0.7,
        "description":  "Estilo cinematográfico tipo película para CogVideoX",
        "use_with":     ["cinematic", "dramatic", "action"],
    },
    {
        "name":         "CogVideoX Anime",
        "source":       "alibaba-pai/CogVideoX-Fun-V1.1-5b-AnimateDiff",
        "category":     "estilo",
        "models":       ["CogVideoX"],
        "action_tags":  ["dancing", "running", "jumping"],
        "style_tags":   ["anime", "animated", "cartoon", "colorful"],
        "trigger":      "",
        "recommended_scale": 0.8,
        "description":  "Estilo anime para CogVideoX",
        "use_with":     ["anime", "colorful", "fantasy"],
    },
    # ── Motion styles ────────────────────────────────────────────────────────
    {
        "name":         "Slow Motion",
        "source":       "Remade/slomo-video-lora",
        "category":     "movimiento",
        "models":       ["LTX-Video", "CogVideoX"],
        "action_tags":  ["running", "jumping", "falling", "exploding", "dancing"],
        "style_tags":   ["slow_motion", "dramatic", "sports"],
        "trigger":      "slow motion",
        "recommended_scale": 0.9,
        "description":  "Efecto slow-motion / cámara lenta",
        "use_with":     ["sports", "action", "nature"],
    },
    {
        "name":         "Walking Natural",
        "source":       "Remade/walking-natural-lora",
        "category":     "movimiento",
        "models":       ["CogVideoX", "LTX-Video"],
        "action_tags":  ["walking", "running", "moving"],
        "style_tags":   ["realistic", "natural", "human"],
        "trigger":      "",
        "recommended_scale": 0.75,
        "description":  "Locomoción humana más natural y fluida",
        "use_with":     ["realistic", "documentary", "human"],
    },
    {
        "name":         "Fluid Hand Motion",
        "source":       "Remade/fluid-hands-lora",
        "category":     "movimiento",
        "models":       ["CogVideoX"],
        "action_tags":  ["waving", "pointing", "grabbing", "clapping", "gesturing"],
        "style_tags":   ["realistic", "gesture"],
        "trigger":      "",
        "recommended_scale": 0.8,
        "description":  "Movimientos de manos más fluidos y naturales",
        "use_with":     ["character_animation", "close_up"],
    },
    {
        "name":         "Face Expression",
        "source":       "Remade/face-expression-lora",
        "category":     "personaje",
        "models":       ["CogVideoX"],
        "action_tags":  ["smiling", "laughing", "crying", "roaring", "talking"],
        "style_tags":   ["portrait", "close_up", "emotional"],
        "trigger":      "",
        "recommended_scale": 0.75,
        "description":  "Expresiones faciales más detalladas",
        "use_with":     ["portrait", "emotional", "drama"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Model family detection
# ─────────────────────────────────────────────────────────────────────────────

def _model_family(model_key: str) -> str:
    if "CogVideoX" in model_key:
        return "CogVideoX"
    if "LTX" in model_key:
        return "LTX-Video"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoRARecommendation:
    name:             str
    source:           str
    category:         str
    recommended_scale:float
    description:      str
    score:            float             # 0.0 – 1.0
    reasons:          list[str] = field(default_factory=list)  # why recommended
    compatible:       bool = True
    frame_hint:       Optional[tuple] = None  # (start_frame, end_frame) suggestion
    curve_hint:       str = "constant"


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Recommender
# ─────────────────────────────────────────────────────────────────────────────

class LoRARecommender:
    """
    Suggests LoRAs based on detected actions, style tags, and model.

    Usage:
        from modules.action_extractor import ActionExtractor
        recommender = LoRARecommender()

        extractor = ActionExtractor()
        actions   = extractor.extract("The eagle slowly flies upward, wings spreading")
        recs      = recommender.recommend(
            actions=actions,
            style_tags=["cinematic", "slow_motion"],
            model_key="LTX-Video (T2V rápido)",
            top_n=5,
        )
        for r in recs:
            print(r.name, r.score, r.reasons)
    """

    def __init__(self, db=None):
        """
        db: optional VideoGenDB instance for historical popularity scoring.
        """
        self._db      = db
        self._history = self._load_history()

    def _load_history(self) -> dict[str, int]:
        """Returns {lora_name: usage_count} from DB if available."""
        if self._db is None:
            return {}
        try:
            records = self._db.get_history(limit=500)
            counts  = {}
            for rec in records:
                for name in rec.loras:
                    counts[name] = counts.get(name, 0) + 1
            return counts
        except Exception:
            return {}

    # ── Main API ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        actions:    list,              # list[ExtractedAction]
        style_tags: list[str],
        model_key:  str,
        total_frames: int = 49,
        top_n:      int = 5,
        user_loras: list[dict] | None = None,
    ) -> list[LoRARecommendation]:
        """
        Returns up to `top_n` LoRA recommendations sorted by score desc.

        Merges built-in DB + user-added LoRAs (from loras/index.json).
        """
        family = _model_family(model_key)

        # Combine knowledge bases
        candidates = list(BUILTIN_LORA_DB)
        if user_loras:
            for ul in user_loras:
                candidates.append({
                    "name":          ul.get("name", ""),
                    "source":        ul.get("source", ""),
                    "category":      ul.get("category", "otro"),
                    "models":        [],   # unknown → always compatible
                    "action_tags":   [],
                    "style_tags":    [],
                    "trigger":       "",
                    "recommended_scale": float(ul.get("scale", 0.8)),
                    "description":   f"LoRA de usuario: {ul.get('name','')}",
                    "use_with":      [],
                })

        recs = []
        for lora_info in candidates:
            rec = self._score(lora_info, actions, style_tags, family, total_frames)
            if rec is not None:
                recs.append(rec)

        # Sort by score desc, then filter top_n
        recs.sort(key=lambda r: r.score, reverse=True)
        return recs[:top_n]

    def recommend_from_prompt(
        self,
        prompt:    str,
        model_key: str,
        total_frames: int = 49,
        top_n:    int = 5,
        user_loras: list[dict] | None = None,
    ) -> list[LoRARecommendation]:
        """Convenience: extract actions from prompt then recommend."""
        from modules.action_extractor import ActionExtractor
        extractor = ActionExtractor()
        actions   = extractor.extract(prompt)
        # Extract style tags from prompt keywords
        style_tags = self._extract_style_tags(prompt)
        return self.recommend(
            actions=actions,
            style_tags=style_tags,
            model_key=model_key,
            total_frames=total_frames,
            top_n=top_n,
            user_loras=user_loras,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        lora_info:    dict,
        actions:      list,
        style_tags:   list[str],
        model_family: str,
        total_frames: int,
    ) -> Optional[LoRARecommendation]:
        name    = lora_info["name"]
        reasons = []
        score   = 0.0

        # 1. Model compatibility check
        supported_models = lora_info.get("models", [])
        if supported_models and not any(m in model_family for m in supported_models):
            compatible = False
            # Still show but penalize heavily
            score -= 0.5
            reasons.append(f"⚠️ Compatible con {supported_models}, no con {model_family}")
        else:
            compatible = True

        # 2. Action relevance (0..0.5)
        action_tags = lora_info.get("action_tags", [])
        detected_verbs = {a.verb for a in actions}
        action_hits = detected_verbs & set(action_tags)
        if action_hits:
            action_score = min(0.5, len(action_hits) * 0.15)
            score += action_score
            reasons.append(f"✅ Acciones detectadas: {', '.join(action_hits)}")

        # 3. Style relevance (0..0.3)
        style_db_tags = lora_info.get("style_tags", [])
        style_hits = set(t.lower() for t in style_tags) & set(style_db_tags)
        if style_hits:
            style_score = min(0.3, len(style_hits) * 0.1)
            score += style_score
            reasons.append(f"🎨 Estilo coincide: {', '.join(style_hits)}")

        # 4. Historical popularity (0..0.2)
        usage = self._history.get(name, 0)
        if usage > 0:
            pop_score = min(0.2, math.log1p(usage) * 0.05)
            score += pop_score
            reasons.append(f"📊 Usado {usage} vez/veces anteriormente")

        # 5. Trigger word bonus (0..0.1)
        trigger = lora_info.get("trigger", "")
        if trigger and any(trigger.lower() in a.raw_text.lower() for a in actions):
            score += 0.1
            reasons.append(f"🔑 Trigger word detectado: '{trigger}'")

        if score <= 0 and not reasons:
            return None   # not relevant enough

        # Frame hint: where this LoRA should be active
        frame_hint = self._suggest_frame_range(lora_info, actions, total_frames)
        curve_hint = self._suggest_curve(lora_info)

        return LoRARecommendation(
            name=name,
            source=lora_info["source"],
            category=lora_info["category"],
            recommended_scale=lora_info["recommended_scale"],
            description=lora_info["description"],
            score=round(max(0.0, min(1.0, score)), 3),
            reasons=reasons,
            compatible=compatible,
            frame_hint=frame_hint,
            curve_hint=curve_hint,
        )

    def _suggest_frame_range(
        self,
        lora_info:    dict,
        actions:      list,
        total_frames: int,
    ) -> Optional[tuple]:
        """
        Suggests which frame range this LoRA should be active in,
        based on when the matching actions occur.
        """
        # Camera LoRAs: typically whole video
        if lora_info["category"] == "cámara":
            return (0, total_frames)

        # Style LoRAs: also whole video
        if lora_info["category"] == "estilo":
            return (0, total_frames)

        # Motion LoRAs: try to narrow to action frames
        action_tags = set(lora_info.get("action_tags", []))
        for a in actions:
            if a.verb in action_tags:
                return (0, total_frames)

        return (0, total_frames)

    def _suggest_curve(self, lora_info: dict) -> str:
        cat = lora_info.get("category", "")
        if cat == "cámara":
            return "fade_in_out"
        if "slow" in lora_info.get("name", "").lower():
            return "constant"
        return "constant"

    def _extract_style_tags(self, prompt: str) -> list[str]:
        p = prompt.lower()
        tags = []
        style_kw = {
            "cinematic": "cinematic", "anime": "anime", "slow": "slow_motion",
            "dramatic": "dramatic", "epic": "epic", "action": "action",
            "portrait": "portrait", "realistic": "realistic",
            "colorful": "colorful", "dark": "dark", "fantasy": "fantasy",
        }
        for kw, tag in style_kw.items():
            if kw in p:
                tags.append(tag)
        return tags

    # ── UI helpers ────────────────────────────────────────────────────────────

    def to_markdown(self, recs: list[LoRARecommendation]) -> str:
        if not recs:
            return "No se encontraron LoRAs relevantes para este prompt."
        lines = [f"### 🎨 {len(recs)} LoRA(s) recomendada(s)\n"]
        for i, r in enumerate(recs, 1):
            compat_icon = "✅" if r.compatible else "⚠️"
            bar = "█" * round(r.score * 10) + "░" * (10 - round(r.score * 10))
            lines.append(
                f"**{i}. {r.name}** {compat_icon}  \n"
                f"Score: `{bar}` {r.score:.2f}  \n"
                f"Fuente: `{r.source}`  \n"
                f"Escala sugerida: `{r.recommended_scale}`  \n"
                f"Frames: `{r.frame_hint[0]}–{r.frame_hint[1]}`  |  "
                f"Curva: `{r.curve_hint}`  \n"
                f"*{r.description}*  \n"
                f"Razones: {' · '.join(r.reasons) if r.reasons else 'coincidencia general'}  \n"
            )
        return "\n".join(lines)

    def to_schedule_json(
        self,
        recs: list[LoRARecommendation],
        total_frames: int = 49,
    ) -> list[dict]:
        """
        Returns a list ready to paste into the LoRA schedule JSON textarea.
        """
        result = []
        for r in recs:
            if not r.compatible:
                continue
            fs, fe = r.frame_hint if r.frame_hint else (0, total_frames)
            result.append({
                "name":        r.name,
                "source":      r.source,
                "scale":       r.recommended_scale,
                "category":    r.category,
                "frame_start": fs,
                "frame_end":   fe,
                "curve":       r.curve_hint,
            })
        return result
