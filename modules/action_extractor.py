"""
modules/action_extractor.py
────────────────────────────
Extracción de acciones estructuradas desde descripciones en lenguaje natural.

Detecta y estructura:
  • Verbo principal  →  acción canónica
  • Sujeto           →  quién realiza la acción
  • Modificadores    →  velocidad · dirección · emoción · intensidad
  • Body-part target →  qué parte del cuerpo está involucrada

No requiere dependencias pesadas (spaCy, NLTK) — usa regexes + diccionarios
curados para funcionar offline en Vast.ai sin descargar modelos NLP extra.

Salida: List[ExtractedAction]  —  lista ordenada de acciones por sujeto
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Verb → canonical action mapping
# Supports English and Spanish
# ─────────────────────────────────────────────────────────────────────────────

VERB_MAP: dict[str, str] = {
    # Locomotion
    "run":      "running",   "runs":    "running",   "running":  "running",
    "correr":   "running",   "corre":   "running",
    "walk":     "walking",   "walks":   "walking",   "walking":  "walking",
    "caminar":  "walking",   "camina":  "walking",
    "jump":     "jumping",   "jumps":   "jumping",   "jumping":  "jumping",
    "saltar":   "jumping",   "salta":   "jumping",
    "fly":      "flying",    "flies":   "flying",    "flying":   "flying",
    "volar":    "flying",    "vuela":   "flying",
    "swim":     "swimming",  "swims":   "swimming",  "swimming": "swimming",
    "nadar":    "swimming",  "nada":    "swimming",
    "fall":     "falling",   "falls":   "falling",   "falling":  "falling",
    "caer":     "falling",   "cae":     "falling",
    "crawl":    "crawling",  "crawls":  "crawling",
    "climb":    "climbing",  "climbs":  "climbing",
    "spin":     "spinning",  "spins":   "spinning",  "spinning": "spinning",
    "girar":    "spinning",  "gira":    "spinning",
    "dance":    "dancing",   "dances":  "dancing",   "dancing":  "dancing",
    "bailar":   "dancing",   "baila":   "dancing",
    "crouch":   "crouching", "crouches":"crouching",
    "agacharse":"crouching",  "agacha":  "crouching",
    "float":    "floating",  "floats":  "floating",  "floating": "floating",
    "flotar":   "floating",  "flota":   "floating",
    "hover":    "hovering",  "hovers":  "hovering",
    "slide":    "sliding",   "slides":  "sliding",

    # Head / face
    "look":     "looking",   "looks":   "looking",   "looking":  "looking",
    "mirar":    "looking",   "mira":    "looking",
    "turn":     "turning",   "turns":   "turning",   "turning":  "turning",
    "girar_cabeza": "head_turning",
    "nod":      "nodding",   "nods":    "nodding",
    "blink":    "blinking",  "blinks":  "blinking",
    "smile":    "smiling",   "smiles":  "smiling",   "smiling":  "smiling",
    "sonreir":  "smiling",   "sonríe":  "smiling",
    "laugh":    "laughing",  "laughs":  "laughing",
    "reír":     "laughing",  "ríe":     "laughing",
    "cry":      "crying",    "cries":   "crying",
    "llorar":   "crying",    "llora":   "crying",
    "roar":     "roaring",   "roars":   "roaring",
    "rugir":    "roaring",   "ruge":    "roaring",

    # Arms / hands
    "wave":     "waving",    "waves":   "waving",
    "point":    "pointing",  "points":  "pointing",
    "señalar":  "pointing",  "señala":  "pointing",
    "grab":     "grabbing",  "grabs":   "grabbing",
    "hold":     "holding",   "holds":   "holding",
    "throw":    "throwing",  "throws":  "throwing",
    "reach":    "reaching",  "reaches": "reaching",
    "push":     "pushing",   "pushes":  "pushing",
    "pull":     "pulling",   "pulls":   "pulling",
    "clap":     "clapping",  "claps":   "clapping",
    "aplaudir": "clapping",

    # Body state
    "stand":    "standing",  "stands":  "standing",
    "sit":      "sitting",   "sits":    "sitting",
    "lie":      "lying_down","lies":    "lying_down",
    "sleep":    "sleeping",  "sleeps":  "sleeping",
    "land":     "landing",   "lands":   "landing",
    "aterrizar":"landing",   "aterriza":"landing",
    "perch":    "perching",  "perches": "perching",
    "stretch":  "stretching","stretches":"stretching",
    "shake":    "shaking",   "shakes":  "shaking",
    "tremble":  "trembling", "trembles":"trembling",
    "temblar":  "trembling", "tiembla": "trembling",
    "bow":      "bowing",    "bows":    "bowing",
    "explode":  "exploding", "explodes":"exploding",
    "burst":    "exploding",

    # Camera (treated as actions on the camera subject)
    "pan":      "camera_pan",
    "zoom":     "camera_zoom",
    "tilt":     "camera_tilt",
    "dolly":    "camera_dolly",
    "orbit":    "camera_orbit",
    "rotate":   "rotating",  "rotates": "rotating",
}

# ─────────────────────────────────────────────────────────────────────────────
# Modifier dictionaries
# ─────────────────────────────────────────────────────────────────────────────

SPEED_MODIFIERS: dict[str, float] = {
    # scale: 0.0=stop, 0.5=slow, 1.0=normal, 1.5=fast, 2.0=very fast
    "slowly":      0.3, "slow":        0.3, "gently":   0.3,
    "lentamente":  0.3, "suavemente":  0.3, "despacio": 0.3,
    "gradually":   0.4,
    "moderately":  0.7, "calmly":      0.5,
    "quickly":     1.5, "fast":        1.5, "rapidly":  1.8,
    "rápidamente": 1.5, "rápido":      1.5, "veloz":    1.7,
    "suddenly":    2.0, "abruptly":    2.0, "instantly": 2.0,
    "de_repente":  2.0, "súbitamente": 2.0,
    "ultra_slow":  0.1, "gracefully":  0.4,
    "frantically": 1.9, "lazily":      0.2,
}

DIRECTION_MODIFIERS: dict[str, tuple] = {
    # (axis, sign)  axis: x=horizontal, y=vertical, z=depth
    "left":         ("x", -1), "right":      ("x",  1),
    "izquierda":    ("x", -1), "derecha":    ("x",  1),
    "up":           ("y",  1), "upward":     ("y",  1), "above":  ("y",  1),
    "arriba":       ("y",  1), "hacia_arriba":("y", 1),
    "down":         ("y", -1), "downward":   ("y", -1), "below":  ("y", -1),
    "abajo":        ("y", -1), "hacia_abajo": ("y",-1),
    "forward":      ("z", -1), "ahead":      ("z", -1), "toward": ("z", -1),
    "adelante":     ("z", -1), "hacia":      ("z", -1),
    "backward":     ("z",  1), "back":       ("z",  1), "away":   ("z",  1),
    "atrás":        ("z",  1),
    "clockwise":    ("r",  1), "counterclockwise": ("r", -1),
}

EMOTION_MODIFIERS: dict[str, str] = {
    "angrily":       "angry",    "furiously":  "furious",   "aggressively": "aggressive",
    "enfadado":      "angry",    "furioso":    "furious",   "agresivo":     "aggressive",
    "happily":       "happy",    "joyfully":   "joyful",    "gleefully":    "happy",
    "felizmente":    "happy",    "alegre":     "happy",
    "sadly":         "sad",      "mournfully": "sad",       "tearfully":    "sad",
    "tristemente":   "sad",      "triste":     "sad",
    "fearfully":     "fearful",  "nervously":  "nervous",   "anxiously":    "anxious",
    "temeroso":      "fearful",  "nervioso":   "nervous",
    "calmly":        "calm",     "peacefully": "calm",      "serenely":     "calm",
    "calmado":       "calm",     "tranquilo":  "calm",      "sereno":       "calm",
    "proudly":       "proud",    "confidently":"confident",
    "orgulloso":     "proud",    "confiado":   "confident",
    "tenderly":      "tender",   "lovingly":   "loving",    "gently":       "gentle",
    "tiernamente":   "tender",   "amorosamente":"loving",
    "desperately":   "desperate","frantically":"frantic",
    "desesperado":   "desperate","frenéticamente":"frantic",
}

INTENSITY_MODIFIERS: dict[str, float] = {
    "slightly":    0.2, "barely":     0.15, "lightly":   0.25,
    "ligeramente": 0.2, "apenas":     0.15,
    "moderately":  0.5, "somewhat":   0.4,
    "fully":       1.0, "completely": 1.0, "entirely":  1.0,
    "totally":     1.0, "deeply":     0.9, "heavily":   0.9,
    "extremely":   1.5, "intensely":  1.4, "violently": 1.6,
    "extremadamente": 1.5, "intensamente": 1.4,
    "half":        0.5, "partially":  0.4, "almost":    0.8,
}

# ─────────────────────────────────────────────────────────────────────────────
# Action → Body-part mapping
# ─────────────────────────────────────────────────────────────────────────────

ACTION_BODY_PARTS: dict[str, list[str]] = {
    "running":      ["legs", "feet", "arms", "torso"],
    "walking":      ["legs", "feet", "arms"],
    "jumping":      ["legs", "feet", "whole_body"],
    "flying":       ["wings", "whole_body"],
    "swimming":     ["arms", "legs", "whole_body"],
    "falling":      ["whole_body"],
    "crawling":     ["arms", "legs", "whole_body"],
    "climbing":     ["arms", "legs", "hands", "feet"],
    "spinning":     ["whole_body", "torso"],
    "dancing":      ["whole_body", "arms", "legs", "hips"],
    "crouching":    ["legs", "knees", "torso"],
    "floating":     ["whole_body"],
    "hovering":     ["whole_body"],
    "sliding":      ["legs", "feet", "whole_body"],
    "looking":      ["head", "eyes", "neck"],
    "turning":      ["head", "neck", "torso"],
    "head_turning": ["head", "neck"],
    "nodding":      ["head", "neck"],
    "blinking":     ["eyes"],
    "smiling":      ["face", "mouth", "cheeks"],
    "laughing":     ["face", "mouth", "torso"],
    "crying":       ["face", "eyes"],
    "roaring":      ["mouth", "head", "torso"],
    "waving":       ["arm", "hand"],
    "pointing":     ["arm", "hand", "index_finger"],
    "grabbing":     ["hand", "fingers", "arm"],
    "holding":      ["hand", "fingers", "arm"],
    "throwing":     ["arm", "hand", "shoulder"],
    "reaching":     ["arm", "hand"],
    "pushing":      ["arm", "hand", "torso"],
    "pulling":      ["arm", "hand", "torso"],
    "clapping":     ["hands", "arms"],
    "standing":     ["legs", "feet", "whole_body"],
    "sitting":      ["hips", "legs", "torso"],
    "lying_down":   ["whole_body"],
    "sleeping":     ["whole_body", "eyes"],
    "landing":      ["legs", "feet", "whole_body"],
    "perching":     ["feet", "legs"],
    "stretching":   ["whole_body", "arms", "legs"],
    "shaking":      ["head", "whole_body"],
    "trembling":    ["whole_body"],
    "bowing":       ["torso", "head", "whole_body"],
    "exploding":    ["whole_body"],
    "rotating":     ["whole_body", "torso"],
    "camera_pan":   ["camera"],
    "camera_zoom":  ["camera", "lens"],
    "camera_tilt":  ["camera"],
    "camera_dolly": ["camera"],
    "camera_orbit": ["camera"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Subject detection patterns
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT_PATTERNS = [
    # Explicit tags from structured prompts
    (r"CHAR_(\d+)", lambda m: f"CHAR_{m.group(1)}"),
    # Common subjects
    (r"\b(lion|tiger|bear|wolf|eagle|bird|dragon|horse|dog|cat|fox)\b", lambda m: m.group(1)),
    (r"\b(man|woman|person|character|figure|warrior|knight|samurai|ninja|wizard|hero)\b", lambda m: m.group(1)),
    (r"\b(robot|android|cyborg|creature|monster|giant|alien)\b", lambda m: m.group(1)),
    (r"\b(camera|cam)\b", lambda m: "camera"),
    # Spanish
    (r"\b(hombre|mujer|persona|guerrero|caballero|bruja|héroe|mago)\b", lambda m: m.group(1)),
    (r"\b(robot|criatura|monstruo|gigante|alien)\b", lambda m: m.group(1)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedAction:
    verb:          str               # canonical action (e.g. "running")
    subject:       str = "unknown"   # who performs the action
    body_parts:    list[str] = field(default_factory=list)
    speed:         float = 1.0       # 0=stop .. 2=very fast
    direction:     Optional[tuple] = None   # (axis, sign) or None
    emotion:       str = ""          # e.g. "angry", "calm"
    intensity:     float = 1.0       # 0=barely .. 1.5=extreme
    raw_text:      str = ""          # original phrase that was parsed
    confidence:    float = 1.0       # 0..1

    def to_prompt_tokens(self) -> list[str]:
        """Convert this action to prompt-ready tokens."""
        tokens = []
        # Speed (check most restrictive first)
        if self.speed < 0.4:
            tokens.append("slowly")
        elif self.speed > 1.8:
            tokens.append("frantically")
        elif self.speed > 1.4:
            tokens.append("rapidly")

        # Emotion
        if self.emotion:
            tokens.append(f"{self.emotion}ly" if not self.emotion.endswith("ly") else self.emotion)

        # Core action
        action_str = self.verb.replace("_", " ")
        tokens.append(action_str)

        # Direction
        if self.direction:
            axis, sign = self.direction
            dir_words = {
                ("x", -1): "to the left", ("x",  1): "to the right",
                ("y",  1): "upward",       ("y", -1): "downward",
                ("z", -1): "forward",      ("z",  1): "backward",
                ("r",  1): "clockwise",    ("r", -1): "counterclockwise",
            }
            tokens.append(dir_words.get((axis, sign), ""))

        # Body parts (only the first 2 most relevant)
        if self.body_parts:
            bp = self.body_parts[:2]
            tokens.append(f"with {' and '.join(bp)}")

        # Intensity qualifier
        if self.intensity > 1.2:
            tokens = ["intensely"] + tokens
        elif self.intensity < 0.3:
            tokens = ["barely"] + tokens

        return [t for t in tokens if t]

    def to_prompt_string(self) -> str:
        return " ".join(self.to_prompt_tokens())

    def __repr__(self) -> str:
        return (f"ExtractedAction(verb={self.verb!r}, subject={self.subject!r}, "
                f"speed={self.speed:.1f}, emotion={self.emotion!r}, "
                f"body={self.body_parts})")


# ─────────────────────────────────────────────────────────────────────────────
# Action Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ActionExtractor:
    """
    Extracts structured actions from natural language descriptions.

    Usage:
        extractor = ActionExtractor()
        actions = extractor.extract(
            "The lion slowly turns its head to the left, roaring aggressively"
        )
        for a in actions:
            print(a)
            print(a.to_prompt_string())
    """

    def __init__(self):
        # Pre-compile verb regex for speed
        self._verb_re = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in sorted(VERB_MAP, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )
        self._subject_res = [
            (re.compile(pat, re.IGNORECASE), fn)
            for pat, fn in SUBJECT_PATTERNS
        ]

    # ── Main entry point ─────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        default_subject: str = "subject",
    ) -> list[ExtractedAction]:
        """
        Returns a list of ExtractedAction objects found in `text`.
        Multiple verbs → multiple actions.
        """
        text = text.strip()
        # Split on known conjunctions to get per-action phrases
        phrases = self._split_phrases(text)
        actions = []

        for phrase in phrases:
            found = self._extract_from_phrase(phrase, default_subject)
            actions.extend(found)

        # Deduplicate same verb+subject
        seen = set()
        unique = []
        for a in actions:
            key = (a.verb, a.subject)
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique

    def extract_from_keyframes(
        self,
        keyframes,            # list[SceneKeyframe]
    ) -> dict[int, list[ExtractedAction]]:
        """
        Returns {keyframe_index: [ExtractedAction, …]}.
        """
        result = {}
        for kf in keyframes:
            subj = "subject"
            if kf.characters:
                subj = kf.characters[0].label if kf.characters else "subject"
            result[kf.index] = self.extract(kf.description, default_subject=subj)
        return result

    def to_prompt_enhancement(self, actions: list[ExtractedAction]) -> str:
        """
        Merges all actions into a rich prompt string.
        Groups by subject for clarity.
        """
        by_subject: dict[str, list[ExtractedAction]] = {}
        for a in actions:
            by_subject.setdefault(a.subject, []).append(a)

        parts = []
        for subj, acts in by_subject.items():
            subj_str = subj if subj != "subject" else ""
            action_strs = [a.to_prompt_string() for a in acts]
            if subj_str and subj_str != "unknown":
                parts.append(f"{subj_str} {' and '.join(action_strs)}")
            else:
                parts.extend(action_strs)

        return ", ".join(parts)

    # ── Internal parsing ─────────────────────────────────────────────────────

    def _split_phrases(self, text: str) -> list[str]:
        """Split text on conjunctions and transition words into phrases."""
        splits = re.split(
            r"\b(and\s+then|then|while|as|meanwhile|simultaneously|"
            r"y\s+luego|luego|mientras|al\s+mismo\s+tiempo|,)\b",
            text, flags=re.IGNORECASE,
        )
        return [s.strip() for s in splits if s.strip() and len(s.strip()) > 3]

    def _extract_from_phrase(
        self,
        phrase: str,
        default_subject: str,
    ) -> list[ExtractedAction]:
        actions = []
        phrase_lower = phrase.lower()

        # Find all verb matches
        for m in self._verb_re.finditer(phrase):
            verb_raw = m.group(1).lower()
            canonical = VERB_MAP.get(verb_raw)
            if not canonical:
                continue

            # Context window: words around the verb (±8 words)
            words = phrase_lower.split()
            try:
                verb_idx = next(
                    i for i, w in enumerate(words)
                    if verb_raw in w
                )
            except StopIteration:
                verb_idx = len(words) // 2

            ctx_start = max(0, verb_idx - 8)
            ctx_end   = min(len(words), verb_idx + 8)
            context   = words[ctx_start:ctx_end]

            action = ExtractedAction(
                verb=canonical,
                subject=self._detect_subject(phrase, default_subject),
                body_parts=ACTION_BODY_PARTS.get(canonical, []),
                speed=self._detect_speed(context),
                direction=self._detect_direction(context),
                emotion=self._detect_emotion(context),
                intensity=self._detect_intensity(context),
                raw_text=phrase[:80],
                confidence=0.9,
            )
            actions.append(action)

        return actions

    def _detect_subject(self, phrase: str, default: str) -> str:
        for pattern_re, fn in self._subject_res:
            m = pattern_re.search(phrase)
            if m:
                return fn(m)
        return default

    def _detect_speed(self, context: list[str]) -> float:
        for word in context:
            word_clean = word.strip(".,!?;:")
            if word_clean in SPEED_MODIFIERS:
                return SPEED_MODIFIERS[word_clean]
        return 1.0

    def _detect_direction(self, context: list[str]) -> Optional[tuple]:
        for word in context:
            word_clean = word.strip(".,!?;:")
            if word_clean in DIRECTION_MODIFIERS:
                return DIRECTION_MODIFIERS[word_clean]
        return None

    def _detect_emotion(self, context: list[str]) -> str:
        for word in context:
            word_clean = word.strip(".,!?;:")
            if word_clean in EMOTION_MODIFIERS:
                return EMOTION_MODIFIERS[word_clean]
        return ""

    def _detect_intensity(self, context: list[str]) -> float:
        for word in context:
            word_clean = word.strip(".,!?;:")
            if word_clean in INTENSITY_MODIFIERS:
                return INTENSITY_MODIFIERS[word_clean]
        return 1.0

    # ── Summary helpers ───────────────────────────────────────────────────────

    def summarize(self, actions: list[ExtractedAction]) -> str:
        if not actions:
            return "No se detectaron acciones."
        lines = [f"🔍 **{len(actions)} acción(es) detectada(s):**"]
        for a in actions:
            speed_tag = (
                "🐢 lento" if a.speed < 0.5 else
                "⚡ rápido" if a.speed > 1.4 else
                "➡️ normal"
            )
            emotion_tag = f" | 😤 {a.emotion}" if a.emotion else ""
            dir_tag = f" | ↗️ {a.direction}" if a.direction else ""
            lines.append(
                f"  • **{a.verb}** (sujeto: *{a.subject}*) — "
                f"{speed_tag}{emotion_tag}{dir_tag}  \n"
                f"    partes del cuerpo: `{', '.join(a.body_parts[:3]) or '—'}`  \n"
                f"    prompt: `{a.to_prompt_string()}`"
            )
        return "\n".join(lines)
