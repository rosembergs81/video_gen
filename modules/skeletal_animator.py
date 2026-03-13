"""
modules/skeletal_animator.py
──────────────────────────────
Sistema de animación esquelética simplificada para generación de prompts.

Proporciona:
  • Esqueletos predefinidos (humano, cuadrúpedo, ave, robot)
  • Poses nombradas (idle, walking, running, jumping, sitting, …)
  • Interpolación de poses entre keyframes
  • Traducción de poses → tokens de prompt descriptivos

No genera imágenes 3D ni usa OpenGL — todo se mantiene en el espacio
de descripción de texto para enriquecer los prompts del diffusion model.
"""

from __future__ import annotations
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from modules.motion_interpolator import MotionInterpolator, EASING_FNS


# ─────────────────────────────────────────────────────────────────────────────
# Joint definitions  (angle in degrees; 0 = neutral anatomical position)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Joint:
    name:     str
    angle_x:  float = 0.0   # pitch  (forward/back flexion)
    angle_y:  float = 0.0   # yaw    (horizontal rotation)
    angle_z:  float = 0.0   # roll   (lateral bending)
    # Position offset relative to parent (normalized units)
    pos_x:    float = 0.0
    pos_y:    float = 0.0
    pos_z:    float = 0.0

    def as_dict(self) -> dict:
        return {
            f"{self.name}_ax": self.angle_x,
            f"{self.name}_ay": self.angle_y,
            f"{self.name}_az": self.angle_z,
            f"{self.name}_px": self.pos_x,
            f"{self.name}_py": self.pos_y,
            f"{self.name}_pz": self.pos_z,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Joint":
        return cls(
            name=name,
            angle_x=d.get(f"{name}_ax", 0.0),
            angle_y=d.get(f"{name}_ay", 0.0),
            angle_z=d.get(f"{name}_az", 0.0),
            pos_x=d.get(f"{name}_px",   0.0),
            pos_y=d.get(f"{name}_py",   0.0),
            pos_z=d.get(f"{name}_pz",   0.0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton  (collection of joints with hierarchy metadata)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Skeleton:
    name:   str
    joints: dict[str, Joint]    # joint_name → Joint
    hierarchy: list[tuple[str, Optional[str]]]  # (joint, parent_or_None)

    def flat_dict(self) -> dict[str, float]:
        """Flat float dict for interpolation."""
        d = {}
        for j in self.joints.values():
            d.update(j.as_dict())
        return d

    @classmethod
    def from_flat(cls, skeleton_name: str, template: "Skeleton", d: dict) -> "Skeleton":
        """Reconstruct skeleton from flat interpolation dict."""
        s = deepcopy(template)
        for jname, joint in s.joints.items():
            joint.angle_x = d.get(f"{jname}_ax", joint.angle_x)
            joint.angle_y = d.get(f"{jname}_ay", joint.angle_y)
            joint.angle_z = d.get(f"{jname}_az", joint.angle_z)
            joint.pos_x   = d.get(f"{jname}_px", joint.pos_x)
            joint.pos_y   = d.get(f"{jname}_py", joint.pos_y)
            joint.pos_z   = d.get(f"{jname}_pz", joint.pos_z)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton factory
# ─────────────────────────────────────────────────────────────────────────────

def _human_skeleton() -> Skeleton:
    joints = {
        "root":          Joint("root"),
        "spine":         Joint("spine"),
        "chest":         Joint("chest"),
        "neck":          Joint("neck"),
        "head":          Joint("head"),
        "shoulder_l":    Joint("shoulder_l"),
        "upper_arm_l":   Joint("upper_arm_l"),
        "forearm_l":     Joint("forearm_l"),
        "hand_l":        Joint("hand_l"),
        "shoulder_r":    Joint("shoulder_r"),
        "upper_arm_r":   Joint("upper_arm_r"),
        "forearm_r":     Joint("forearm_r"),
        "hand_r":        Joint("hand_r"),
        "hip_l":         Joint("hip_l"),
        "thigh_l":       Joint("thigh_l"),
        "shin_l":        Joint("shin_l"),
        "foot_l":        Joint("foot_l"),
        "hip_r":         Joint("hip_r"),
        "thigh_r":       Joint("thigh_r"),
        "shin_r":        Joint("shin_r"),
        "foot_r":        Joint("foot_r"),
    }
    hierarchy = [
        ("root", None), ("spine", "root"), ("chest", "spine"),
        ("neck", "chest"), ("head", "neck"),
        ("shoulder_l", "chest"), ("upper_arm_l", "shoulder_l"),
        ("forearm_l", "upper_arm_l"), ("hand_l", "forearm_l"),
        ("shoulder_r", "chest"), ("upper_arm_r", "shoulder_r"),
        ("forearm_r", "upper_arm_r"), ("hand_r", "forearm_r"),
        ("hip_l", "root"), ("thigh_l", "hip_l"),
        ("shin_l", "thigh_l"), ("foot_l", "shin_l"),
        ("hip_r", "root"), ("thigh_r", "hip_r"),
        ("shin_r", "thigh_r"), ("foot_r", "shin_r"),
    ]
    return Skeleton("human", joints, hierarchy)


def _quadruped_skeleton() -> Skeleton:
    joints = {
        "root":      Joint("root"),
        "spine":     Joint("spine"),
        "neck":      Joint("neck"),
        "head":      Joint("head"),
        "tail":      Joint("tail"),
        "fl_shoulder": Joint("fl_shoulder"),
        "fl_leg":      Joint("fl_leg"),
        "fl_paw":      Joint("fl_paw"),
        "fr_shoulder": Joint("fr_shoulder"),
        "fr_leg":      Joint("fr_leg"),
        "fr_paw":      Joint("fr_paw"),
        "rl_hip":      Joint("rl_hip"),
        "rl_leg":      Joint("rl_leg"),
        "rl_paw":      Joint("rl_paw"),
        "rr_hip":      Joint("rr_hip"),
        "rr_leg":      Joint("rr_leg"),
        "rr_paw":      Joint("rr_paw"),
    }
    hierarchy = [
        ("root", None), ("spine", "root"), ("neck", "spine"), ("head", "neck"),
        ("tail", "root"),
        ("fl_shoulder", "spine"), ("fl_leg", "fl_shoulder"), ("fl_paw", "fl_leg"),
        ("fr_shoulder", "spine"), ("fr_leg", "fr_shoulder"), ("fr_paw", "fr_leg"),
        ("rl_hip", "root"), ("rl_leg", "rl_hip"), ("rl_paw", "rl_leg"),
        ("rr_hip", "root"), ("rr_leg", "rr_hip"), ("rr_paw", "rr_leg"),
    ]
    return Skeleton("quadruped", joints, hierarchy)


def _bird_skeleton() -> Skeleton:
    joints = {
        "root":       Joint("root"),
        "body":       Joint("body"),
        "neck":       Joint("neck"),
        "head":       Joint("head"),
        "beak":       Joint("beak"),
        "wing_l":     Joint("wing_l"),
        "wingtip_l":  Joint("wingtip_l"),
        "wing_r":     Joint("wing_r"),
        "wingtip_r":  Joint("wingtip_r"),
        "tail":       Joint("tail"),
        "leg_l":      Joint("leg_l"),
        "claw_l":     Joint("claw_l"),
        "leg_r":      Joint("leg_r"),
        "claw_r":     Joint("claw_r"),
    }
    hierarchy = [
        ("root", None), ("body", "root"), ("neck", "body"), ("head", "neck"),
        ("beak", "head"), ("tail", "body"),
        ("wing_l", "body"), ("wingtip_l", "wing_l"),
        ("wing_r", "body"), ("wingtip_r", "wing_r"),
        ("leg_l", "body"), ("claw_l", "leg_l"),
        ("leg_r", "body"), ("claw_r", "leg_r"),
    ]
    return Skeleton("bird", joints, hierarchy)


SKELETON_TEMPLATES = {
    "human":      _human_skeleton,
    "quadruped":  _quadruped_skeleton,
    "bird":       _bird_skeleton,
}


# ─────────────────────────────────────────────────────────────────────────────
# Named poses  (flat dicts: joint_name_ax/ay/az → degrees)
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_POSES: dict[str, dict] = {
    "idle": {
        "spine_ax": 0, "head_ax": 0,
        "upper_arm_l_az": 15, "upper_arm_r_az": -15,
        "thigh_l_ax": 0,  "thigh_r_ax": 0,
    },
    "walking": {
        "spine_ax": 5,
        "upper_arm_l_ax":  30, "upper_arm_r_ax": -30,
        "forearm_l_ax":   -20, "forearm_r_ax":   20,
        "thigh_l_ax":      25, "thigh_r_ax":    -25,
        "shin_l_ax":      -20, "shin_r_ax":      15,
        "foot_l_ax":       10, "foot_r_ax":     -10,
    },
    "running": {
        "spine_ax": 10,
        "upper_arm_l_ax":  60, "upper_arm_r_ax": -60,
        "forearm_l_ax":   -45, "forearm_r_ax":   45,
        "thigh_l_ax":      50, "thigh_r_ax":    -50,
        "shin_l_ax":      -40, "shin_r_ax":      35,
    },
    "jumping_peak": {
        "spine_ax": -5, "head_ax": 5,
        "upper_arm_l_ax": 150, "upper_arm_r_ax": 150,
        "thigh_l_ax": -20, "thigh_r_ax": -20,
        "shin_l_ax":  -30, "shin_r_ax":  -30,
    },
    "crouching": {
        "spine_ax": 20,
        "thigh_l_ax":  80, "thigh_r_ax":  80,
        "shin_l_ax": -80, "shin_r_ax": -80,
        "upper_arm_l_az": 30, "upper_arm_r_az": -30,
    },
    "sitting": {
        "spine_ax": 5,
        "thigh_l_ax": 90, "thigh_r_ax": 90,
        "shin_l_ax":  -90, "shin_r_ax": -90,
    },
    "lying_down": {
        "spine_ax": 90, "head_ax": -90,
        "upper_arm_l_az": 90, "upper_arm_r_az": -90,
        "thigh_l_ax": 0, "thigh_r_ax": 0,
    },
    "pointing": {
        "spine_ay": 15,
        "upper_arm_r_ax": 70, "upper_arm_r_az": -10,
        "forearm_r_ax": -10,
        "hand_r_ax": -5,
    },
    "waving": {
        "upper_arm_r_ax": 90, "upper_arm_r_az": -15,
        "forearm_r_ax": 45,
        "hand_r_az": 30,
    },
    "arms_raised": {
        "upper_arm_l_ax": 160, "upper_arm_r_ax": 160,
        "forearm_l_ax": -10,  "forearm_r_ax": -10,
    },
    "bowing": {
        "spine_ax": 60, "head_ax": 30,
        "upper_arm_l_ax": 20, "upper_arm_r_ax": 20,
    },
    "dancing": {
        "spine_ax": 10, "spine_ay": 15,
        "upper_arm_l_ax": 70,  "upper_arm_l_az": 30,
        "upper_arm_r_ax": 50,  "upper_arm_r_az": -20,
        "thigh_l_ax": 30, "thigh_r_ax": -10,
    },
}

QUADRUPED_POSES: dict[str, dict] = {
    "idle": {
        "head_ax": 0, "tail_ax": 10,
        "fl_leg_ax": 0, "fr_leg_ax": 0,
    },
    "walking": {
        "head_ax": 5,
        "fl_leg_ax": 30, "fr_leg_ax": -20,
        "rl_leg_ax": -20, "rr_leg_ax": 30,
        "tail_ax": 15,
    },
    "running": {
        "spine_ax": 15, "head_ax": 10,
        "fl_leg_ax": 60,  "fr_leg_ax": -50,
        "rl_leg_ax": -50, "rr_leg_ax": 60,
        "tail_ax": 30,
    },
    "roaring": {
        "head_ax": -30, "head_ay": 5,
        "neck_ax": -20,
        "fl_leg_ax": -10, "fr_leg_ax": -10,
        "tail_ax": 40,
    },
    "jumping": {
        "spine_ax": 20,
        "fl_leg_ax": 60,  "fr_leg_ax": 60,
        "rl_leg_ax": -60, "rr_leg_ax": -60,
        "tail_ax": 20,
    },
    "crouching": {
        "spine_ax": 10,
        "fl_leg_ax": 40, "fr_leg_ax": 40,
        "rl_leg_ax": 40, "rr_leg_ax": 40,
        "head_ax": 5,
    },
}

BIRD_POSES: dict[str, dict] = {
    "perched": {
        "body_ax": 5,
        "wing_l_ax": 10, "wing_r_ax": 10,
        "wingtip_l_ax": 15, "wingtip_r_ax": 15,
        "tail_ax": -10,
    },
    "flying": {
        "body_ax": -15,
        "wing_l_az": -60, "wing_r_az": 60,
        "wingtip_l_az": -30, "wingtip_r_az": 30,
        "tail_ax": 10,
    },
    "wings_up": {
        "wing_l_az": -90, "wing_r_az": 90,
        "wingtip_l_az": -45, "wingtip_r_az": 45,
    },
    "wings_down": {
        "wing_l_az": 30, "wing_r_az": -30,
        "wingtip_l_az": 10, "wingtip_r_az": -10,
    },
    "landing": {
        "body_ax": 30,
        "wing_l_az": -45, "wing_r_az": 45,
        "leg_l_ax": -30, "leg_r_ax": -30,
        "tail_ax": -20,
    },
}

POSE_LIBRARY: dict[str, dict[str, dict]] = {
    "human":     HUMAN_POSES,
    "quadruped": QUADRUPED_POSES,
    "bird":      BIRD_POSES,
}


# ─────────────────────────────────────────────────────────────────────────────
# Skeletal Animator
# ─────────────────────────────────────────────────────────────────────────────

class SkeletalAnimator:
    """
    Manages pose interpolation and prompt generation for a skeleton type.

    Usage:
        anim = SkeletalAnimator("human")
        prompt = anim.interpolate_to_prompt("idle", "running", t=0.5)
        print(prompt)

        sequence = anim.animation_sequence(
            ["idle", "running", "jumping_peak", "crouching"],
            total_frames=49,
        )
        for frame, desc in sequence:
            print(frame, desc)
    """

    def __init__(self, skeleton_type: str = "human"):
        if skeleton_type not in SKELETON_TEMPLATES:
            raise ValueError(f"skeleton_type must be one of {list(SKELETON_TEMPLATES)}")
        self.skeleton_type = skeleton_type
        self.template      = SKELETON_TEMPLATES[skeleton_type]()
        self.poses         = POSE_LIBRARY.get(skeleton_type, {})
        self._interp       = MotionInterpolator()

    # ── Core interpolation ────────────────────────────────────────────────────

    def interpolate(
        self,
        pose_a_name: str,
        pose_b_name: str,
        t: float,
        easing: str = "ease_in_out",
    ) -> dict:
        """Returns interpolated flat joint dict at t=[0,1]."""
        a = self._get_pose_flat(pose_a_name)
        b = self._get_pose_flat(pose_b_name)
        return self._interp.lerp_pose(a, b, t, easing)

    def interpolate_sequence(
        self,
        pose_names: list[str],
        t: float,
        easing: str = "ease_in_out",
    ) -> dict:
        """Interpolates across N named poses at t=[0,1]."""
        flat_poses = [self._get_pose_flat(n) for n in pose_names]
        return self._interp.lerp_poses_sequence(flat_poses, t, easing)

    # ── Prompt generation ─────────────────────────────────────────────────────

    def pose_to_prompt(self, pose_dict: dict) -> str:
        """Converts joint angles to descriptive prompt tokens."""
        tokens = []

        def angle(key, default=0.0):
            return pose_dict.get(key, default)

        # ── HEAD ────────────────────────────────────────────────────────────
        if self.skeleton_type in ("human", "quadruped", "bird"):
            head_ax = angle("head_ax")
            head_ay = angle("head_ay")
            if abs(head_ax) > 20:
                tokens.append("head tilted " + ("upward" if head_ax < 0 else "downward"))
            if abs(head_ay) > 15:
                tokens.append("head turned to the " + ("left" if head_ay < 0 else "right"))

        # ── SPINE / BODY ────────────────────────────────────────────────────
        if self.skeleton_type == "human":
            spine_ax = angle("spine_ax")
            if spine_ax > 30:
                tokens.append("torso bending forward")
            elif spine_ax > 10:
                tokens.append("torso leaning forward")
            elif spine_ax < -10:
                tokens.append("torso arching back")

            # ── ARMS ─────────────────────────────────────────────────────
            arm_l = angle("upper_arm_l_ax")
            arm_r = angle("upper_arm_r_ax")
            if arm_l > 120 and arm_r > 120:
                tokens.append("both arms raised overhead")
            elif arm_r > 60:
                tokens.append("right arm raised")
            elif arm_l > 60:
                tokens.append("left arm raised")
            if angle("forearm_r_ax") < -30:
                tokens.append("right elbow bent")

            # ── LEGS ─────────────────────────────────────────────────────
            thigh_l = angle("thigh_l_ax")
            thigh_r = angle("thigh_r_ax")
            if thigh_l > 60 and angle("shin_l_ax") < -60:
                tokens.append("left leg fully bent, crouching")
            elif abs(thigh_l - thigh_r) > 25:
                tokens.append("legs in stride position")

        elif self.skeleton_type == "quadruped":
            tail_ax = angle("tail_ax")
            if tail_ax > 30:
                tokens.append("tail raised high")
            fl_ax = angle("fl_leg_ax")
            rl_ax = angle("rl_leg_ax")
            if abs(fl_ax - rl_ax) > 40:
                tokens.append("legs in galloping stride")

        elif self.skeleton_type == "bird":
            wing_az = angle("wing_l_az")
            if wing_az < -50:
                tokens.append("wings fully spread upward")
            elif wing_az < -20:
                tokens.append("wings partially spread")
            elif wing_az > 20:
                tokens.append("wings folded downward")

        return ", ".join(tokens) if tokens else "neutral stance"

    def interpolate_to_prompt(
        self,
        pose_a: str,
        pose_b: str,
        t: float,
        easing: str = "ease_in_out",
    ) -> str:
        """One-liner: pose names + t → prompt string."""
        interpolated = self.interpolate(pose_a, pose_b, t, easing)
        return self.pose_to_prompt(interpolated)

    # ── Animation sequence ────────────────────────────────────────────────────

    def animation_sequence(
        self,
        pose_names: list[str],
        total_frames: int = 49,
        easing: str = "ease_in_out",
    ) -> list[tuple[int, str]]:
        """
        Returns [(frame_number, prompt_description), …] for each frame.
        Useful for per-frame prompt injection.
        """
        result = []
        for frame in range(total_frames):
            t = frame / max(total_frames - 1, 1)
            flat = self.interpolate_sequence(pose_names, t, easing)
            desc = self.pose_to_prompt(flat)
            result.append((frame, desc))
        return result

    def keyframe_prompts(
        self,
        pose_names: list[str],
        total_frames: int = 49,
    ) -> list[tuple[int, str, str]]:
        """
        Returns (frame, pose_name, prompt_description) for each named pose,
        evenly distributed. Useful for prompt preview.
        """
        result = []
        n = len(pose_names)
        for i, pname in enumerate(pose_names):
            frame = round(i / max(n - 1, 1) * (total_frames - 1))
            flat  = self._get_pose_flat(pname)
            desc  = self.pose_to_prompt(flat)
            result.append((frame, pname, desc))
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_pose_flat(self, pose_name: str) -> dict:
        """Returns flat dict for a named pose, falling back to idle/zero."""
        neutral = {k: 0.0 for k in self.template.flat_dict()}
        named   = self.poses.get(pose_name, {})
        return {**neutral, **named}

    def list_poses(self) -> list[str]:
        return list(self.poses.keys())

    def available_skeletons(self) -> list[str]:
        return list(SKELETON_TEMPLATES.keys())

    # ── UI summary ────────────────────────────────────────────────────────────

    def sequence_markdown(
        self,
        pose_names: list[str],
        total_frames: int = 49,
    ) -> str:
        kfs = self.keyframe_prompts(pose_names, total_frames)
        lines = [f"**Animación esquelética** ({self.skeleton_type}) — {total_frames} frames\n"]
        for frame, pname, desc in kfs:
            lines.append(f"- **Frame {frame:3d}** `{pname}` → _{desc}_")
        return "\n".join(lines)
