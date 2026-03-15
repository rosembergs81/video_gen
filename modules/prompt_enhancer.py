"""
modules/prompt_enhancer.py
───────────────────────────
Motor de mejora de prompts con inyección de contexto cinematográfico.
Transforma prompts básicos en descripciones ricas y detalladas,
similar a como Grok/Meta AI internamente enriquecen los inputs.
"""

from __future__ import annotations
import re


# ─────────────────────────────────────────────────────────────────────────────
# Context enhancement library
# ─────────────────────────────────────────────────────────────────────────────

LIGHTING_ENHANCERS = {
    "golden_hour":  "warm golden sunlight, long soft shadows, lens flare, hazy atmosphere, bokeh background",
    "blue_hour":    "cool blue twilight, ambient glow, soft diffused light",
    "midday_sun":   "harsh direct sunlight, sharp shadows, high contrast, bleached colors",
    "overcast":     "even diffused light, no harsh shadows, muted palette, dramatic clouds",
    "neon_night":   "neon glow, high contrast, saturated electric colors, wet reflections, cyberpunk atmosphere",
    "candlelight":  "warm flickering candlelight, deep shadows, intimate orange glow",
    "moonlight":    "cool silver moonlight, long blue shadows, ethereal atmosphere",
    "studio":       "professional three-point studio lighting, clean background, sharp details",
    "backlit":      "strong backlight, rim lighting, silhouette effect, volumetric rays",
    "magic_hour":   "magical twilight, pink and orange sky, dreamy warm tones",
}

CINEMATOGRAPHY_ENHANCERS = {
    "slow_mo":        "ultra slow motion, high speed photography, motion blur trails, time suspended",
    "steadicam":      "smooth flowing steadicam movement, professional cinematography, gliding",
    "handheld":       "intimate handheld feel, slight natural shake, documentary realism",
    "aerial":         "breathtaking aerial perspective, bird's eye view, vast scale",
    "macro":          "extreme close-up, macro detail, shallow depth of field, bokeh",
    "wide_angle":     "expansive wide angle, dramatic perspective distortion, grand sense of scale",
    "telephoto":      "telephoto compression, isolated subject, blurred background, intimate framing",
    "dutch_angle":    "unnerving dutch angle tilt, tension-building composition",
    "rack_focus":     "rack focus transition, background dissolving to blur, foreground sharp",
    "tracking_shot":  "dynamic tracking shot following subject, kinetic energy",
}

QUALITY_BOOSTERS = {
    "photorealistic": "photorealistic, 8K resolution, hyper-detailed, physically accurate",
    "cinematic":      "cinematic quality, film grain, anamorphic lens flare, color graded",
    "concept_art":    "concept art quality, highly detailed illustration, professional render",
    "hyperrealistic": "hyperrealistic, ultra-sharp, studio quality, award-winning photography",
    "4k_film":        "4K resolution, professional film quality, RAW footage look",
}

ATMOSPHERE_ENHANCERS = {
    "fog":            "atmospheric fog, volumetric mist, depth through haze",
    "rain":           "rain falling, wet surfaces reflecting, rain drops on lens",
    "snow":           "snowflakes falling, winter atmosphere, cold color palette",
    "dust":           "dust particles floating, golden dust motes in light beams",
    "smoke":          "drifting smoke, hazy atmosphere, volumetric light through smoke",
    "underwater":     "underwater scene, caustic light patterns, bubbles rising",
    "fire":           "fire and embers, dynamic warm light, smoke rising",
    "wind":           "wind-blown hair and fabric, dynamic movement, atmospheric",
}

GENRE_TEMPLATES = {
    "action":     {
        "motion":   "fast-paced action, dynamic movement, explosive energy",
        "camera":   "tracking shot, rapid cutting energy, kinetic camera",
        "quality":  "high contrast, punchy colors, sharp details",
    },
    "romance":    {
        "motion":   "soft gentle movement, intimate close-ups, tender gestures",
        "camera":   "slow romantic dolly, soft focus edges, warm framing",
        "quality":  "warm color grade, soft skin tones, dreamy bokeh",
    },
    "horror":     {
        "motion":   "slow creeping movement, sudden jolts, unsettling stillness",
        "camera":   "low angle, dutch tilt, extreme shadow",
        "quality":  "desaturated, high contrast shadows, cold blue tones",
    },
    "documentary":{
        "motion":   "natural authentic movement, unposed realism",
        "camera":   "handheld follow, observational framing",
        "quality":  "natural color, realistic grain, honest lighting",
    },
    "fantasy":    {
        "motion":   "magical flowing movement, ethereal floating, graceful arcs",
        "camera":   "majestic wide angles, slow orbit, epic scale",
        "quality":  "vibrant saturated colors, magical light effects, wonder",
    },
    "scifi":      {
        "motion":   "precise mechanical movement, advanced technology interaction",
        "camera":   "clinical tracking, wide establishing, technical detail shots",
        "quality":  "clean futuristic aesthetic, neon accents, sleek surfaces",
    },
}

# Subject detection → automatic context injection
SUBJECT_CONTEXTS = {
    r"\b(beach|ocean|sea|waves)\b":        "ocean spray, golden sand, surf",
    r"\b(forest|woods|trees)\b":           "dappled light through leaves, rustling foliage",
    r"\b(city|urban|street|downtown)\b":   "urban atmosphere, ambient city noise implied",
    r"\b(mountain|peak|summit|cliff)\b":   "vast mountain atmosphere, thin air, epic scale",
    r"\b(space|galaxy|stars|cosmos)\b":    "infinite cosmic scale, starfield, void of space",
    r"\b(desert|sand|dunes|arid)\b":       "shimmering heat haze, vast emptiness, dry heat",
    r"\b(rain|storm|thunder|lightning)\b": "dramatic storm atmosphere, wet environment",
    r"\b(snow|winter|ice|frozen)\b":       "winter stillness, cold breath visible, frost",
    r"\b(fire|flame|burning|lava)\b":      "dynamic fire light, heat distortion, embers",
    r"\b(underwater|ocean floor|coral)\b": "underwater light caustics, slow float",
}


# ─────────────────────────────────────────────────────────────────────────────
# Negative prompt library
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_PRESETS = {
    "standard": (
        "blurry, low quality, low resolution, poorly drawn, bad anatomy, "
        "deformed, distorted, disfigured, watermark, text, signature, "
        "overexposed, underexposed, out of focus, grainy, "
        "extra fingers, mutated hands, missing fingers, extra limbs, "
        "fused fingers, too many fingers, bad hands, missing arms, "
        "long neck, cross-eyed, mutation, ugly, cropped"
    ),
    "realistic": (
        "cartoon, anime, illustration, painting, drawing, digital art, "
        "blurry, low quality, artifacts, watermark, text, unrealistic, "
        "extra fingers, mutated hands, deformed body, bad anatomy, "
        "missing limbs, extra limbs, fused fingers, bad proportions"
    ),
    "cinematic": (
        "amateur, low budget, poor lighting, bad composition, blurry, "
        "noise, grain, flickering, interlaced, low resolution, watermark, "
        "extra fingers, mutated hands, deformed, bad anatomy, ugly"
    ),
    "animation": (
        "photorealistic, low quality, choppy animation, missing frames, "
        "inconsistent style, watermark, text, poor detail, "
        "extra fingers, mutated hands, deformed, bad anatomy"
    ),
    "minimal": "blurry, low quality, watermark, text, bad anatomy, deformed",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Enhancer
# ─────────────────────────────────────────────────────────────────────────────

class PromptEnhancer:
    """
    Enriches a base prompt with cinematographic context, lighting,
    atmosphere, quality tags, and genre-specific enhancements.

    Usage:
        enhancer = PromptEnhancer()
        result = enhancer.enhance(
            base="A lion roaring in the savanna",
            lighting="golden_hour",
            cinematography="slow_mo",
            quality="cinematic",
            atmosphere="dust",
            genre="documentary",
            auto_context=True,
        )
    """

    def enhance(
        self,
        base: str,
        lighting:       str | None = None,
        cinematography: str | None = None,
        quality:        str | None = None,
        atmosphere:     str | None = None,
        genre:          str | None = None,
        extra_tags:     list[str] | None = None,
        auto_context:   bool = True,
        max_enhancement_tokens: int = 80,
    ) -> str:
        """
        Enhance a base prompt with cinematographic context.
        
        IMPORTANT: CogVideoX processes ~226 tokens max. We limit enhancements
        to avoid overloading the model with conflicting directives.
        """
        base_clean = base.strip().rstrip(",. ")
        parts = [base_clean]
        enhancement_parts = []

        # Auto-inject based on subject matter (max 1 to avoid noise)
        if auto_context:
            ctx = self._detect_context(base)
            if ctx:
                enhancement_parts.append(ctx)

        # Quality booster (highest priority enhancement)
        if quality and quality in QUALITY_BOOSTERS:
            enhancement_parts.append(QUALITY_BOOSTERS[quality])

        # Lighting (second priority)
        if lighting and lighting in LIGHTING_ENHANCERS:
            enhancement_parts.append(LIGHTING_ENHANCERS[lighting])

        # Genre template (only add motion tag, not all 3 sub-tags)
        if genre and genre in GENRE_TEMPLATES:
            tmpl = GENRE_TEMPLATES[genre]
            # Only add the most impactful tag (motion) to avoid overloading
            if "motion" in tmpl:
                enhancement_parts.append(tmpl["motion"])

        # Cinematography (lower priority — skip if we already have enough)
        if cinematography and cinematography in CINEMATOGRAPHY_ENHANCERS:
            enhancement_parts.append(CINEMATOGRAPHY_ENHANCERS[cinematography])

        # Atmosphere (lowest priority)
        if atmosphere and atmosphere in ATMOSPHERE_ENHANCERS:
            enhancement_parts.append(ATMOSPHERE_ENHANCERS[atmosphere])

        # Extra free-form tags
        if extra_tags:
            enhancement_parts.extend([t.strip() for t in extra_tags if t.strip()])

        # Enforce token budget for enhancements
        total_enhancement = ", ".join(enhancement_parts)
        enhancement_tokens = len(total_enhancement.split())
        if enhancement_tokens > max_enhancement_tokens:
            # Truncate enhancement to budget
            words = total_enhancement.split()[:max_enhancement_tokens]
            total_enhancement = " ".join(words)

        if total_enhancement:
            parts.append(total_enhancement)

        return ", ".join(p for p in parts if p)

    def _detect_context(self, text: str) -> str:
        """Auto-detect scene context from subject keywords."""
        text_lower = text.lower()
        detected = []
        for pattern, context in SUBJECT_CONTEXTS.items():
            if re.search(pattern, text_lower):
                detected.append(context)
        return ", ".join(detected[:1])  # at most 1 auto-context to avoid noise

    def get_negative(self, preset: str = "standard") -> str:
        return NEGATIVE_PRESETS.get(preset, NEGATIVE_PRESETS["standard"])

    def suggest_enhancements(self, base: str) -> dict:
        """
        Suggests applicable enhancements based on prompt analysis.
        Returns a dict of {category: [suggestions]}.
        """
        text = base.lower()
        suggestions = {}

        # Lighting suggestions
        if any(w in text for w in ["sunset", "sunrise", "dusk", "dawn", "atardecer"]):
            suggestions["lighting"] = ["golden_hour", "magic_hour"]
        elif any(w in text for w in ["night", "noche", "dark", "oscuro"]):
            suggestions["lighting"] = ["neon_night", "moonlight"]
        elif any(w in text for w in ["indoor", "interior", "room"]):
            suggestions["lighting"] = ["studio", "candlelight"]

        # Genre suggestions
        if any(w in text for w in ["fight", "battle", "action", "pelea", "combate"]):
            suggestions["genre"] = ["action"]
        elif any(w in text for w in ["love", "kiss", "romantic", "amor", "beso"]):
            suggestions["genre"] = ["romance"]
        elif any(w in text for w in ["scary", "horror", "terror", "ghost", "fantasma"]):
            suggestions["genre"] = ["horror"]
        elif any(w in text for w in ["future", "robot", "space", "cyber", "futuro"]):
            suggestions["genre"] = ["scifi"]
        elif any(w in text for w in ["magic", "dragon", "wizard", "elf", "magia"]):
            suggestions["genre"] = ["fantasy"]

        # Atmosphere
        if any(w in text for w in ["rain", "lluvia", "storm", "tormenta"]):
            suggestions["atmosphere"] = ["rain"]
        elif any(w in text for w in ["fire", "fuego", "lava", "flame"]):
            suggestions["atmosphere"] = ["fire"]
        elif any(w in text for w in ["fog", "mist", "niebla"]):
            suggestions["atmosphere"] = ["fog"]

        return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# Convenience exports for UI dropdowns
# ─────────────────────────────────────────────────────────────────────────────

LIGHTING_OPTIONS      = ["—"] + list(LIGHTING_ENHANCERS.keys())
CINEMATOGRAPHY_OPTIONS= ["—"] + list(CINEMATOGRAPHY_ENHANCERS.keys())
QUALITY_OPTIONS       = ["—"] + list(QUALITY_BOOSTERS.keys())
ATMOSPHERE_OPTIONS    = ["—"] + list(ATMOSPHERE_ENHANCERS.keys())
GENRE_OPTIONS         = ["—"] + list(GENRE_TEMPLATES.keys())
NEGATIVE_PRESETS_LIST = list(NEGATIVE_PRESETS.keys())
