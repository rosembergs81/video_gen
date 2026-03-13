"""
modules/gesture_templates.py
─────────────────────────────
Biblioteca de gestos de manos, expresiones faciales y animaciones
de cuerpo completo. Cada gesto produce tokens de prompt que describen
la posición y movimiento de partes del cuerpo.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GestureFrame:
    """A single phase of a gesture animation."""
    t: float           # normalized time within the gesture (0.0–1.0)
    hands:  str = ""   # prompt tokens for hands
    face:   str = ""   # prompt tokens for face
    body:   str = ""   # prompt tokens for body/torso
    eyes:   str = ""
    motion: str = ""   # overall motion descriptor


@dataclass
class GestureTemplate:
    name: str
    category: str              # "hand", "facial", "full_body", "interaction"
    duration_hint: float = 1.0 # relative duration (1.0 = normal speed)
    loop: bool = False
    frames: list[GestureFrame] = field(default_factory=list)
    transition: str = "smooth" # "smooth" | "snap" | "eased"

    def prompt_at(self, t: float) -> str:
        """Returns a combined prompt string at normalized time t."""
        if not self.frames:
            return ""
        # Find surrounding frames
        prev = self.frames[0]
        for f in self.frames:
            if f.t <= t:
                prev = f
            else:
                break
        parts = [s for s in [prev.hands, prev.face, prev.body, prev.motion] if s]
        return ", ".join(parts)

    def full_description(self) -> str:
        """Returns a static description merging all phases."""
        all_parts = set()
        for f in self.frames:
            for s in [f.hands, f.face, f.body, f.motion]:
                if s:
                    all_parts.add(s)
        return ", ".join(sorted(all_parts))


# ─────────────────────────────────────────────────────────────────────────────
# Gesture library
# ─────────────────────────────────────────────────────────────────────────────

GESTURE_TEMPLATES: dict[str, GestureTemplate] = {

    # ── Hand gestures ────────────────────────────────────────────────────────

    "kiss": GestureTemplate(
        name="kiss", category="hand",
        frames=[
            GestureFrame(0.0,
                hands="both hands raised near lips",
                face="lips pursed in kiss shape",
                body="slight forward lean"),
            GestureFrame(0.5,
                hands="fingers together touching lips",
                face="mouth rounded, blowing kiss",
                body="torso leaning forward"),
            GestureFrame(0.8,
                hands="hand extending outward, fingers spreading",
                face="soft smile, eyes half-closed",
                motion="blowing kiss motion"),
            GestureFrame(1.0,
                hands="arm extended, palm open toward camera",
                face="warm smile",
                motion="kiss sent, hand lowering gently"),
        ],
    ),

    "wave": GestureTemplate(
        name="wave", category="hand", loop=True,
        frames=[
            GestureFrame(0.0,
                hands="right hand raised to shoulder height",
                body="slight head tilt accompanying wave"),
            GestureFrame(0.33,
                hands="right hand waving to the right, fingers together",
                motion="hand moving laterally"),
            GestureFrame(0.66,
                hands="right hand waving back to center",
                motion="hand returning"),
            GestureFrame(1.0,
                hands="right hand at shoulder height again"),
        ],
    ),

    "point": GestureTemplate(
        name="point", category="hand",
        frames=[
            GestureFrame(0.0,
                hands="arm at rest by side",
                body="neutral standing"),
            GestureFrame(0.4,
                hands="arm raising, index finger extending",
                body="torso rotating toward direction"),
            GestureFrame(1.0,
                hands="arm fully extended, index finger pointing ahead",
                body="leaning slightly forward",
                motion="confident pointing gesture"),
        ],
    ),

    "thumbs_up": GestureTemplate(
        name="thumbs_up", category="hand",
        frames=[
            GestureFrame(0.0, hands="hand closed in fist"),
            GestureFrame(0.5, hands="fist raising, thumb extending upward"),
            GestureFrame(1.0,
                hands="thumb fully raised, arm slightly extended",
                face="confident smile",
                motion="approval gesture"),
        ],
    ),

    "clap": GestureTemplate(
        name="clap", category="hand", loop=True, duration_hint=0.5,
        frames=[
            GestureFrame(0.0,  hands="both hands apart at chest level"),
            GestureFrame(0.5,  hands="both palms clapping together",
                         motion="hands clapping together"),
            GestureFrame(1.0,  hands="hands apart again"),
        ],
    ),

    "reach_up": GestureTemplate(
        name="reach_up", category="hand",
        frames=[
            GestureFrame(0.0, hands="arms at sides", body="standing straight"),
            GestureFrame(0.6,
                hands="arms rising above head, fingers reaching upward",
                body="slight arch in back, rising on toes",
                motion="reaching upward"),
            GestureFrame(1.0,
                hands="arms fully extended overhead, fingertips pointing up",
                body="stretched tall",
                motion="fully extended upward reach"),
        ],
    ),

    # ── Facial expressions ───────────────────────────────────────────────────

    "smile": GestureTemplate(
        name="smile", category="facial",
        frames=[
            GestureFrame(0.0, face="neutral expression"),
            GestureFrame(0.4, face="corners of mouth lifting slightly"),
            GestureFrame(1.0, face="warm genuine smile, cheeks lifted, eyes crinkling",
                         motion="joyful expression"),
        ],
    ),

    "surprised": GestureTemplate(
        name="surprised", category="facial",
        frames=[
            GestureFrame(0.0, face="neutral calm expression"),
            GestureFrame(0.2,
                face="eyes widening, eyebrows raising sharply",
                body="slight backward lean",
                motion="sudden surprise"),
            GestureFrame(0.6,
                face="eyes wide open, mouth open in surprise",
                body="hands raising slightly in shock"),
            GestureFrame(1.0,
                face="open-mouthed surprise, eyebrows fully raised"),
        ],
    ),

    "thinking": GestureTemplate(
        name="thinking", category="facial",
        frames=[
            GestureFrame(0.0, face="neutral"),
            GestureFrame(0.5,
                hands="right hand raised, index finger touching chin or temple",
                face="eyes looking upward, slight brow furrow",
                body="head tilted slightly",
                motion="contemplative thinking pose"),
            GestureFrame(1.0,
                hands="finger resting on chin",
                face="thoughtful expression, eyes distant",
                motion="deep in thought"),
        ],
    ),

    # ── Full-body gestures ───────────────────────────────────────────────────

    "bow": GestureTemplate(
        name="bow", category="full_body",
        frames=[
            GestureFrame(0.0, body="standing upright, arms at sides"),
            GestureFrame(0.5,
                body="torso bending forward at waist, head lowering",
                motion="respectful bow"),
            GestureFrame(1.0,
                body="deeply bowed, head near 45° angle",
                motion="deep bow of respect"),
        ],
    ),

    "turn_and_look": GestureTemplate(
        name="turn_and_look", category="full_body",
        frames=[
            GestureFrame(0.0, body="facing forward", face="looking straight ahead"),
            GestureFrame(0.5,
                body="body rotating, shoulders turning",
                face="head beginning to turn",
                motion="smooth body rotation"),
            GestureFrame(1.0,
                body="fully turned to the side or back",
                face="head turned, eyes looking over shoulder",
                motion="complete turn"),
        ],
    ),

    "dance_step": GestureTemplate(
        name="dance_step", category="full_body", loop=True,
        frames=[
            GestureFrame(0.0,
                body="weight on left foot, arms positioned gracefully",
                motion="dance beginning"),
            GestureFrame(0.25,
                body="stepping right, arm extending",
                motion="fluid dance movement"),
            GestureFrame(0.5,
                body="weight shifting, arms arcing",
                motion="rhythmic body movement"),
            GestureFrame(0.75,
                body="spinning motion, hair and clothes following",
                motion="pirouette beginning"),
            GestureFrame(1.0,
                body="completing spin, returning to center",
                motion="graceful dance step completion"),
        ],
    ),

    "jump": GestureTemplate(
        name="jump", category="full_body",
        frames=[
            GestureFrame(0.0, body="knees slightly bent, arms back — crouching"),
            GestureFrame(0.3,
                body="legs pushing off, arms swinging forward and up",
                motion="launching upward"),
            GestureFrame(0.6,
                body="airborne, legs trailing, arms raised",
                motion="peak of jump, suspended in air"),
            GestureFrame(0.85,
                body="descending, legs bending for landing",
                motion="falling from jump"),
            GestureFrame(1.0,
                body="landing, knees absorbing impact",
                motion="landing from jump"),
        ],
    ),

    # ── Interaction gestures ─────────────────────────────────────────────────

    "handshake": GestureTemplate(
        name="handshake", category="interaction",
        frames=[
            GestureFrame(0.0,  hands="right arm extending outward toward other person"),
            GestureFrame(0.4,  hands="right hand meeting the other's hand"),
            GestureFrame(0.7,  hands="hands firmly clasped, shaking up and down",
                         motion="firm handshake"),
            GestureFrame(1.0,  hands="hands releasing, arms returning to sides"),
        ],
    ),

    "hug": GestureTemplate(
        name="hug", category="interaction",
        frames=[
            GestureFrame(0.0, body="arms open, stepping forward", motion="approaching"),
            GestureFrame(0.5, body="arms wrapping around the other person",
                         motion="embracing"),
            GestureFrame(0.8, body="fully embraced, holding close", motion="warm hug"),
            GestureFrame(1.0, body="slowly releasing, stepping back"),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_gesture(name: str) -> GestureTemplate | None:
    return GESTURE_TEMPLATES.get(name)

def list_gestures(category: str | None = None) -> list[str]:
    if category:
        return [k for k, v in GESTURE_TEMPLATES.items() if v.category == category]
    return list(GESTURE_TEMPLATES.keys())

def gesture_to_prompt(name: str, t: float = 0.5) -> str:
    """Quick helper: gesture name + time → prompt string."""
    g = get_gesture(name)
    return g.prompt_at(t) if g else ""

def build_gesture_sequence(
    gestures: list[tuple[str, float, float]],  # (name, t_start, t_end)
    total_frames: int,
    current_frame: int,
) -> str:
    """
    Given a list of (gesture_name, t_start, t_end) pairs
    and the current frame, returns the active gesture prompt.
    """
    t = current_frame / max(total_frames - 1, 1)
    for name, t_start, t_end in gestures:
        if t_start <= t <= t_end:
            local_t = (t - t_start) / max(t_end - t_start, 1e-6)
            g = get_gesture(name)
            if g:
                return g.prompt_at(local_t)
    return ""


GESTURE_CATEGORIES = {
    "✋ Manos":        list_gestures("hand"),
    "😊 Faciales":    list_gestures("facial"),
    "🕺 Cuerpo completo": list_gestures("full_body"),
    "🤝 Interacción": list_gestures("interaction"),
}
