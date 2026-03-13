"""
modules/temporal_validator.py  (v2)
─────────────────────────────────────
Validación temporal-física completa de secuencias de video.

Integra:
  • Reglas semánticas de transición (imposibilidad lógica)
  • Reglas de momentum / física cinemática  ← NUEVO
  • PhysicsValidator  (aceleración, ground, camera jitter)  ← NUEVO
  • ActionExtractor  (tokenización semántica rica)  ← NUEVO
  • Score compuesto con breakdown por categoría

Uso rápido:
    validator = TemporalCoherenceValidator()
    report    = validator.validate_from_prompt(
        "A lion starts roaring → bird flies → bird lands on lion", frames=49
    )
    print(report.to_markdown())

Uso completo:
    validator = TemporalCoherenceValidator(fps=24, total_frames=49)
    keyframes = SceneParser().parse(prompt)
    report    = validator.validate_keyframes(keyframes)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from modules.action_extractor  import ActionExtractor, ExtractedAction
from modules.physics_validator import PhysicsValidator, PhysicsReport, PhysicsIssue


# ─────────────────────────────────────────────────────────────────────────────
# Semantic transition rules
# ─────────────────────────────────────────────────────────────────────────────

# (action_a, action_b) → min_frames required between them
TRANSITION_COSTS: dict[tuple[str, str], int] = {
    # Basic locomotion transitions
    ("jumping",        "sitting"):          12,
    ("jumping",        "sleeping"):         30,
    ("running",        "sleeping"):         24,
    ("running",        "sitting"):           8,
    ("crouching",      "jumping"):           4,
    ("lying_down",     "running"):          16,
    ("lying_down",     "jumping"):          20,
    ("lying_down",     "standing"):         10,
    ("swimming",       "running"):          20,
    ("swimming",       "standing"):         15,
    # Speed-state transitions
    ("exploding",      "standing"):         20,
    ("exploding",      "floating"):         25,
    ("flying_fast",    "landing"):          15,
    ("falling",        "floating"):         10,
    ("falling",        "standing"):          8,
    ("spinning",       "standing"):          6,
    # Body state transitions
    ("sleeping",       "running"):          30,
    ("sleeping",       "jumping"):          35,
    ("sleeping",       "standing"):         12,
    ("sitting",        "running"):           8,
    ("sitting",        "jumping"):          10,
    # Animal-specific
    ("roaring",        "sleeping"):         20,
    ("roaring",        "sitting"):          10,
    # Interaction
    ("grabbing",       "throwing"):          4,
    ("landing",        "running"):           6,
    ("landing",        "jumping"):           8,
}

# Mutually exclusive within the same frame
EXCLUSIVE_PAIRS: list[frozenset] = [
    frozenset({"running",     "sleeping"}),
    frozenset({"jumping",     "sitting"}),
    frozenset({"swimming",    "burning"}),
    frozenset({"frozen",      "running"}),
    frozenset({"lying_down",  "jumping"}),
    frozenset({"exploding",   "floating"}),
    frozenset({"flying",      "crawling"}),
    frozenset({"standing",    "lying_down"}),
]

# Prerequisites: action B requires A to have occurred first
PREREQUISITES: dict[str, list[str]] = {
    "landing":      ["jumping", "falling", "flying"],
    "catching":     ["throwing", "falling"],
    "ducking":      ["running", "standing"],
    "blocking":     ["standing"],
    "throwing":     ["grabbing", "holding"],
    "perching":     ["flying", "landing"],
}

# ── Momentum rules (NEW) ─────────────────────────────────────────────────────

# Actions with high inertia — they resist sudden stops
HIGH_INERTIA_ACTIONS = {
    "running", "flying", "falling", "exploding", "spinning", "sliding"
}

# Minimum deceleration frames before stopping from high-inertia actions
DECELERATION_FRAMES: dict[str, int] = {
    "running":   6,
    "flying":    8,
    "falling":   4,
    "exploding": 12,
    "spinning":  5,
    "sliding":   4,
}

# Actions that require a build-up phase
BUILD_UP_REQUIRED: dict[str, tuple[str, int]] = {
    # action → (required_predecessor, min_frames)
    "running":   ("walking", 3),
    "jumping":   ("crouching", 2),
    "sprinting": ("running", 4),
    "exploding": ("standing", 1),
}

# ─────────────────────────────────────────────────────────────────────────────
# Validation result models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity:    str                  # "error" | "warning" | "suggestion"
    frame_start: int
    frame_end:   int
    action_a:    str
    action_b:    str
    message:     str
    fix:         Optional[str] = None
    category:    str = "semantic"     # "semantic" | "physics" | "momentum"


@dataclass
class ValidationReport:
    is_valid:          bool
    issues:            list[ValidationIssue]
    score:             float          # 0.0 – 1.0  (composite)
    score_semantic:    float = 1.0    # 0.0 – 1.0
    score_physics:     float = 1.0    # 0.0 – 1.0
    score_momentum:    float = 1.0    # 0.0 – 1.0
    physics_report:    Optional[PhysicsReport] = None
    detected_actions:  list[ExtractedAction] = field(default_factory=list)
    summary:           str = ""

    def errors(self)      -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]
    def warnings(self)    -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]
    def suggestions(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "suggestion"]

    def to_markdown(self) -> str:
        lines = []

        # Score bar
        def bar(score):
            filled = round(score * 10)
            return "█" * filled + "░" * (10 - filled)

        lines.append(
            f"**Score compuesto: {self.score:.2f}** | "
            f"Semántico: `{bar(self.score_semantic)}` {self.score_semantic:.2f} | "
            f"Física: `{bar(self.score_physics)}` {self.score_physics:.2f} | "
            f"Momentum: `{bar(self.score_momentum)}` {self.score_momentum:.2f}\n"
        )

        if not self.issues:
            lines.append("✅ Sin problemas detectados. Secuencia coherente.")
        else:
            cats = {"semantic": "🧠 Semántico", "physics": "⚙️ Física", "momentum": "🏃 Momentum"}
            by_cat: dict[str, list] = {}
            for iss in self.issues:
                by_cat.setdefault(iss.category, []).append(iss)

            for cat_key, iss_list in by_cat.items():
                lines.append(f"\n**{cats.get(cat_key, cat_key)}**")
                for iss in iss_list:
                    icon = "🔴" if iss.severity == "error" else \
                           "🟡" if iss.severity == "warning" else "🔵"
                    lines.append(
                        f"{icon} **frames {iss.frame_start}→{iss.frame_end}:** {iss.message}"
                    )
                    if iss.fix:
                        lines.append(f"   💡 *Fix:* {iss.fix}")

        # Physics breakdown
        if self.physics_report and self.physics_report.issues:
            lines.append(f"\n**⚙️ Detalle de física:**\n{self.physics_report.to_markdown()}")

        # Detected actions
        if self.detected_actions:
            lines.append(
                f"\n**🔍 Acciones detectadas:** "
                + ", ".join(f"`{a.verb}`" for a in self.detected_actions[:8])
            )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Coherence Validator  (v2)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCoherenceValidator:
    """
    Comprehensive validator integrating:
      • Semantic transition rules
      • Momentum / inertia rules
      • PhysicsValidator (acceleration, ground, jitter)
      • ActionExtractor (rich NLP tokenization)

    Usage:
        v = TemporalCoherenceValidator()
        r = v.validate_from_prompt("lion runs then jumps then sleeps", frames=49)
        print(r.to_markdown())
    """

    def __init__(
        self,
        fps:          int  = 24,
        total_frames: int  = 49,
        subject_type: str  = "default",
    ):
        self.fps           = fps
        self.total_frames  = total_frames
        self.subject_type  = subject_type
        self._extractor    = ActionExtractor()
        self._physics      = PhysicsValidator(fps=fps, subject_type=subject_type)
        self._custom_rules: list[tuple] = []

    # ── Custom rules ──────────────────────────────────────────────────────────

    def add_rule(
        self,
        action_a:      str,
        action_b:      str,
        min_gap_frames:int,
        message:       str = "",
    ):
        self._custom_rules.append((action_a, action_b, min_gap_frames, message))

    # ── Entry points ──────────────────────────────────────────────────────────

    def validate_from_prompt(
        self,
        prompt: str,
        frames: int | None = None,
    ) -> ValidationReport:
        """Quick validation directly from a raw prompt string."""
        from modules.scene_parser import SceneParser
        total = frames or self.total_frames
        kfs   = SceneParser().parse(prompt, total_frames=total)
        return self.validate_keyframes(kfs, total_frames=total)

    def validate_keyframes(
        self,
        keyframes,
        total_frames: int | None = None,
    ) -> ValidationReport:
        """Validates a list of SceneKeyframe objects."""
        total = total_frames or self.total_frames
        sequence = [
            (round(kf.t_start * total), kf.description)
            for kf in keyframes
        ]
        return self.validate_sequence(sequence, total_frames=total)

    def validate_sequence(
        self,
        sequence:     list[tuple[int, str]],
        total_frames: int | None = None,
    ) -> ValidationReport:
        """
        Full validation pipeline.
        sequence: [(frame_idx, description_str), …]
        """
        total = total_frames or self.total_frames
        issues: list[ValidationIssue] = []

        # ── Extract actions from all keyframes ────────────────────────────────
        all_actions: list[ExtractedAction] = []
        kf_actions:  list[list[ExtractedAction]] = []

        for _, desc in sequence:
            acts = self._extractor.extract(desc)
            all_actions.extend(acts)
            kf_actions.append(acts)

        # ── 1. Semantic transition checks ─────────────────────────────────────
        sem_issues = []
        for i in range(len(sequence) - 1):
            f_a, desc_a = sequence[i]
            f_b, desc_b = sequence[i + 1]
            gap = f_b - f_a

            tokens_a = {a.verb for a in kf_actions[i]}
            tokens_b = {a.verb for a in kf_actions[i + 1]}

            # Transition cost checks
            for tok_a in tokens_a:
                for tok_b in tokens_b:
                    cost = TRANSITION_COSTS.get((tok_a, tok_b), 0)
                    if cost and gap < cost:
                        sem_issues.append(ValidationIssue(
                            severity="error",
                            frame_start=f_a, frame_end=f_b,
                            action_a=tok_a, action_b=tok_b,
                            category="semantic",
                            message=(
                                f"Transición imposible: `{tok_a}` → `{tok_b}` "
                                f"en {gap} frames (mínimo: {cost})"
                            ),
                            fix=(
                                f"Añade {cost - gap} frames o una acción intermedia "
                                f"entre frames {f_a} y {f_b}."
                            ),
                        ))

            # Exclusive pair checks
            combined = tokens_a & tokens_b if gap == 0 else set()
            for pair in EXCLUSIVE_PAIRS:
                if pair.issubset(tokens_a | tokens_b) and gap < 3:
                    a, b = tuple(pair)
                    sem_issues.append(ValidationIssue(
                        severity="error",
                        frame_start=f_a, frame_end=f_b,
                        action_a=a, action_b=b,
                        category="semantic",
                        message=f"`{a}` y `{b}` son mutuamente excluyentes.",
                        fix="Separa en keyframes con mayor distancia temporal.",
                    ))

        # Prerequisite checks
        seen_verbs: set[str] = set()
        for i, (f, _) in enumerate(sequence):
            for act in kf_actions[i]:
                prereqs = PREREQUISITES.get(act.verb, [])
                for pre in prereqs:
                    if pre not in seen_verbs:
                        sem_issues.append(ValidationIssue(
                            severity="warning",
                            frame_start=f, frame_end=f,
                            action_a=pre, action_b=act.verb,
                            category="semantic",
                            message=(
                                f"`{act.verb}` requiere `{pre}` primero, "
                                f"pero no fue detectado."
                            ),
                            fix=f"Añade un keyframe de `{pre}` antes del frame {f}.",
                        ))
            seen_verbs.update(a.verb for a in kf_actions[i])

        # Custom rules
        for rule_a, rule_b, min_gap, msg in self._custom_rules:
            for i in range(len(sequence) - 1):
                f_a, desc_a = sequence[i]
                f_b, desc_b = sequence[i + 1]
                acts_a = {a.verb for a in kf_actions[i]}
                acts_b = {a.verb for a in kf_actions[i + 1]}
                if rule_a in acts_a and rule_b in acts_b and (f_b - f_a) < min_gap:
                    sem_issues.append(ValidationIssue(
                        severity="error",
                        frame_start=f_a, frame_end=f_b,
                        action_a=rule_a, action_b=rule_b,
                        category="semantic",
                        message=msg or f"Regla personalizada: `{rule_a}`→`{rule_b}` muy rápido.",
                        fix=f"Necesitas al menos {min_gap} frames de transición.",
                    ))

        issues.extend(sem_issues)

        # ── 2. Momentum / inertia checks ─────────────────────────────────────
        mom_issues = []
        for i in range(len(sequence) - 1):
            f_a, _ = sequence[i]
            f_b, _ = sequence[i + 1]
            gap    = f_b - f_a

            acts_a = {a.verb for a in kf_actions[i]}
            acts_b = {a.verb for a in kf_actions[i + 1]}

            # High-inertia → sudden stop
            for hi_action in HIGH_INERTIA_ACTIONS & acts_a:
                stop_states = {"standing", "sitting", "lying_down", "idle"}
                if acts_b & stop_states:
                    min_decel = DECELERATION_FRAMES.get(hi_action, 5)
                    if gap < min_decel:
                        mom_issues.append(ValidationIssue(
                            severity="warning",
                            frame_start=f_a, frame_end=f_b,
                            action_a=hi_action,
                            action_b=list(acts_b & stop_states)[0],
                            category="momentum",
                            message=(
                                f"Parada brusca desde `{hi_action}`: necesitas "
                                f"≥{min_decel} frames de desaceleración."
                            ),
                            fix=(
                                f"Añade una acción intermedia de transición "
                                f"(e.g. 'slowing down', 'decelerating') en frame {f_a + gap//2}."
                            ),
                        ))

            # Build-up checks
            for action, (prereq, min_build) in BUILD_UP_REQUIRED.items():
                if action in acts_b and prereq not in {a.verb for a in all_actions[:i+1]}:
                    mom_issues.append(ValidationIssue(
                        severity="suggestion",
                        frame_start=f_a, frame_end=f_b,
                        action_a=prereq, action_b=action,
                        category="momentum",
                        message=(
                            f"`{action}` se ve más natural con un build-up de "
                            f"`{prereq}` previo ({min_build}+ frames)."
                        ),
                        fix=f"Considera añadir `{prereq}` antes de `{action}`.",
                    ))

            # Speed coherence via action speed modifiers
            for act in kf_actions[i]:
                if act.speed > 1.6:
                    # Check if next action is also fast or has deceleration
                    next_speeds = [a.speed for a in kf_actions[i + 1]]
                    if next_speeds and max(next_speeds) < 0.5 and gap < 8:
                        mom_issues.append(ValidationIssue(
                            severity="warning",
                            frame_start=f_a, frame_end=f_b,
                            action_a=act.verb, action_b="slow_action",
                            category="momentum",
                            message=(
                                f"Cambio de velocidad muy brusco: `{act.verb}` (rápido) "
                                f"→ acción lenta en {gap} frames."
                            ),
                            fix="Añade frames de transición de velocidad o usa 'gradually slowing'.",
                        ))

        issues.extend(mom_issues)

        # ── 3. Physics validation ─────────────────────────────────────────────
        # Build proper SimpleKeyframe objects that physics_validator can iterate
        from dataclasses import make_dataclass
        SimpleKF = make_dataclass("SimpleKF", ["t_start", "t_end", "description"])
        phys_keyframes = [
            SimpleKF(
                t_start=seq[0] / max(total, 1),
                t_end=(seq[0] + 1) / max(total, 1),
                description=seq[1],
            )
            for seq in sequence
        ]
        phys_report = (
            self._physics.validate_from_keyframes(
                phys_keyframes,
                total_frames=total,
                subject_type=self.subject_type,
            )
            if phys_keyframes else None
        )

        # Convert physics issues to ValidationIssue
        if phys_report:
            for pi in phys_report.issues:
                issues.append(ValidationIssue(
                    severity=pi.severity,
                    frame_start=pi.frame,
                    frame_end=pi.frame,
                    action_a=pi.issue_type,
                    action_b="",
                    category="physics",
                    message=pi.message,
                    fix=pi.fix,
                ))

        # ── Scores ───────────────────────────────────────────────────────────
        sem_errs  = sum(1 for i in sem_issues  if i.severity == "error")
        sem_warns = sum(1 for i in sem_issues  if i.severity == "warning")
        mom_errs  = sum(1 for i in mom_issues  if i.severity == "error")
        mom_warns = sum(1 for i in mom_issues  if i.severity == "warning")
        phy_score = phys_report.score if phys_report else 1.0

        score_sem = max(0.0, 1.0 - sem_errs * 0.3 - sem_warns * 0.1)
        score_mom = max(0.0, 1.0 - mom_errs * 0.25 - mom_warns * 0.1)
        score_phy = phy_score
        score_composite = (score_sem * 0.45 + score_mom * 0.25 + score_phy * 0.30)

        return ValidationReport(
            is_valid=all(i.severity != "error" for i in issues),
            issues=issues,
            score=round(score_composite, 2),
            score_semantic=round(score_sem, 2),
            score_physics=round(score_phy, 2),
            score_momentum=round(score_mom, 2),
            physics_report=phys_report,
            detected_actions=all_actions,
            summary=(
                f"{sem_errs + mom_errs} errores, "
                f"{sem_warns + mom_warns} advertencias"
            ),
        )

    # ── Static quick-check ────────────────────────────────────────────────────

    @staticmethod
    def quick_check(prompt: str) -> list[str]:
        """
        Lightweight check returning warning strings.
        No frame data needed — works on raw prompt text.
        """
        warnings = []
        p = prompt.lower()

        pairs = [
            (["jumping", "jump"], ["sleeping", "sleep"], "jumping + sleeping"),
            (["running", "run"],  ["floating", "float"], "running + floating"),
            (["explod"],          ["calm", "standing", "still"], "explosión + calma inmediata"),
            (["swimming", "swim"],["fire", "burning", "flame"], "natación + fuego"),
        ]
        for kws_a, kws_b, label in pairs:
            if any(k in p for k in kws_a) and any(k in p for k in kws_b):
                warnings.append(f"⚠️ Combinación posiblemente incoherente: {label}")

        if len(prompt.split(",")) > 14:
            warnings.append(
                "⚠️ Prompt muy largo (>14 tags). "
                "Considera dividirlo con `→` para usar el parser de escenas."
            )

        # Momentum quick-check
        hi_inertia = ["running fast", "flying fast", "exploding", "spinning fast"]
        stops       = ["standing still", "sitting", "sleeping", "frozen"]
        if any(h in p for h in hi_inertia) and any(s in p for s in stops):
            warnings.append(
                "⚠️ Cambio de alta velocidad a reposo detectado — "
                "añade frames de desaceleración para mayor realismo."
            )

        return warnings

    # ── Convenience: full report from raw text ────────────────────────────────

    def full_analysis(self, prompt: str) -> str:
        """
        Returns a complete markdown analysis for display in the UI.
        Includes: detected actions, keyframes, validation report.
        """
        from modules.scene_parser import SceneParser

        keyframes = SceneParser().parse(prompt, total_frames=self.total_frames)
        report    = self.validate_keyframes(keyframes)
        quick     = self.quick_check(prompt)

        kf_md = "\n".join(
            f"- **KF {i+1}** ({kf.t_start:.2f}–{kf.t_end:.2f}): "
            f"{kf.description[:70]}{'…' if len(kf.description) > 70 else ''}"
            for i, kf in enumerate(keyframes)
        )

        actions_md = ""
        if report.detected_actions:
            actions_md = "\n**Acciones detectadas:**\n" + "\n".join(
                f"  • `{a.verb}` — sujeto: *{a.subject}*, "
                f"velocidad: {a.speed:.1f}x"
                + (f", emoción: {a.emotion}" if a.emotion else "")
                + (f", dirección: {a.direction}" if a.direction else "")
                for a in report.detected_actions[:10]
            )

        quick_md = ""
        if quick:
            quick_md = "\n**⚡ Advertencias rápidas:**\n" + "\n".join(quick)

        return (
            f"### Secuencia detectada ({len(keyframes)} keyframes):\n{kf_md}\n\n"
            f"{actions_md}\n\n"
            f"### Validación:\n{report.to_markdown()}"
            f"{quick_md}"
        )
