"""
modules/motion_interpolator.py
───────────────────────────────
Interpola suavemente entre poses y posiciones de cámara usando:
  • Lerp lineal para valores simples
  • Catmull-Rom splines para trayectorias de cámara
  • Ease-in / ease-out para transiciones naturales
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Easing functions
# ─────────────────────────────────────────────────────────────────────────────

def ease_in_out(t: float) -> float:
    """Smooth step: slow start, fast middle, slow end."""
    return t * t * (3 - 2 * t)

def ease_in(t: float) -> float:
    return t * t

def ease_out(t: float) -> float:
    return 1 - (1 - t) ** 2

def ease_in_cubic(t: float) -> float:
    return t ** 3

def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3

EASING_FNS = {
    "linear":       lambda t: t,
    "ease_in_out":  ease_in_out,
    "ease_in":      ease_in,
    "ease_out":     ease_out,
    "ease_cubic_in":  ease_in_cubic,
    "ease_cubic_out": ease_out_cubic,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pose interpolation
# ─────────────────────────────────────────────────────────────────────────────

class MotionInterpolator:
    """
    Interpola entre dos poses (dicts de float) con easing configurable.

    Usage:
        interp = MotionInterpolator()
        pose_a = {"head_x": 0.0, "head_y": 0.0, "arm_angle": 0.0}
        pose_b = {"head_x": 0.3, "head_y": 0.1, "arm_angle": 45.0}
        mid    = interp.lerp_pose(pose_a, pose_b, t=0.5)
    """

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def lerp_pose(
        pose_a: dict[str, float],
        pose_b: dict[str, float],
        t: float,
        easing: str = "ease_in_out",
    ) -> dict[str, float]:
        """
        t = 0.0 → pose_a,  t = 1.0 → pose_b
        Keys present in pose_b but not pose_a are passed through unchanged.
        """
        fn = EASING_FNS.get(easing, ease_in_out)
        te = fn(max(0.0, min(1.0, t)))

        result = {}
        all_keys = set(pose_a) | set(pose_b)
        for k in all_keys:
            va = pose_a.get(k, 0.0)
            vb = pose_b.get(k, va)
            result[k] = va + (vb - va) * te
        return result

    @staticmethod
    def lerp_poses_sequence(
        poses: list[dict[str, float]],
        t: float,
        easing: str = "ease_in_out",
    ) -> dict[str, float]:
        """
        Interpolates across N poses evenly distributed along t=[0,1].
        """
        if not poses:
            return {}
        if len(poses) == 1:
            return poses[0]

        n = len(poses) - 1
        segment = min(int(t * n), n - 1)
        local_t = (t * n) - segment
        return MotionInterpolator.lerp_pose(
            poses[segment], poses[segment + 1], local_t, easing
        )

    @staticmethod
    def pose_to_prompt_tokens(pose: dict[str, float]) -> list[str]:
        """
        Converts numeric pose values to natural-language tokens.
        Convention: keys ending in _angle are degrees, others 0..1 normalized.
        """
        tokens = []
        for k, v in pose.items():
            if "angle" in k:
                deg = round(v)
                tokens.append(f"{k.replace('_', ' ')} at {deg}°")
            else:
                level = (
                    "fully" if v > 0.85 else
                    "mostly" if v > 0.60 else
                    "halfway" if v > 0.35 else
                    "slightly"
                )
                tokens.append(f"{k.replace('_', ' ')} {level}")
        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Camera Path with Catmull-Rom splines
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraKeyframe:
    frame: int
    position: tuple = (0.0, 0.0, 0.0)   # (x, y, z)  world units
    rotation: tuple = (0.0, 0.0, 0.0)   # (pitch, yaw, roll) degrees
    fov: float = 50.0                    # field of view degrees
    easing: str = "ease_in_out"

    def as_dict(self) -> dict:
        return {
            "frame":    self.frame,
            "pos_x":    self.position[0],
            "pos_y":    self.position[1],
            "pos_z":    self.position[2],
            "rot_pitch":self.rotation[0],
            "rot_yaw":  self.rotation[1],
            "rot_roll": self.rotation[2],
            "fov":      self.fov,
        }


def _catmull_rom(p0, p1, p2, p3, t: float) -> float:
    """Catmull-Rom spline through p1→p2, parameterized by t=[0,1]."""
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t*t +
        (-p0 + 3*p1 - 3*p2 + p3) * t*t*t
    )


class CameraPath:
    """
    Defines a smooth camera trajectory through a set of CameraKeyframes.
    Uses Catmull-Rom splines for positions and rotations.

    Usage:
        path = CameraPath()
        path.add_keyframe(CameraKeyframe(0,   position=(0,0,5), fov=50))
        path.add_keyframe(CameraKeyframe(24,  position=(0,0,3), fov=45))  # zoom in
        path.add_keyframe(CameraKeyframe(48,  position=(1,0,3), fov=45))  # pan right
        pose = path.get_pose(frame=12)
    """

    def __init__(self, keyframes: list[CameraKeyframe] | None = None):
        self.keyframes: list[CameraKeyframe] = sorted(
            keyframes or [], key=lambda k: k.frame
        )

    def add_keyframe(self, kf: CameraKeyframe):
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.frame)

    def get_pose(self, frame: int) -> CameraKeyframe:
        """Returns interpolated camera pose at given frame."""
        kfs = self.keyframes
        if not kfs:
            return CameraKeyframe(frame)
        if len(kfs) == 1 or frame <= kfs[0].frame:
            return kfs[0]
        if frame >= kfs[-1].frame:
            return kfs[-1]

        # Find surrounding keyframes
        for i in range(len(kfs) - 1):
            if kfs[i].frame <= frame <= kfs[i+1].frame:
                t = (frame - kfs[i].frame) / max(
                    kfs[i+1].frame - kfs[i].frame, 1
                )
                fn = EASING_FNS.get(kfs[i].easing, ease_in_out)
                te = fn(t)
                return self._interpolate_kfs(i, te)
        return kfs[-1]

    def _interpolate_kfs(self, idx: int, t: float) -> CameraKeyframe:
        kfs = self.keyframes
        n   = len(kfs)

        # Ghost points for Catmull-Rom
        p0 = kfs[max(0, idx - 1)].as_dict()
        p1 = kfs[idx].as_dict()
        p2 = kfs[min(n - 1, idx + 1)].as_dict()
        p3 = kfs[min(n - 1, idx + 2)].as_dict()

        def interp(k):
            return _catmull_rom(p0[k], p1[k], p2[k], p3[k], t)

        pos = (interp("pos_x"), interp("pos_y"), interp("pos_z"))
        rot = (interp("rot_pitch"), interp("rot_yaw"), interp("rot_roll"))
        fov  = interp("fov")

        return CameraKeyframe(
            frame=round(p1["frame"] + (p2["frame"] - p1["frame"]) * t),
            position=pos, rotation=rot, fov=fov,
        )

    def to_prompt_description(self, frame: int, total_frames: int) -> str:
        """
        Returns a natural-language camera description for the given frame.
        """
        pose = self.get_pose(frame)

        parts = []
        # FOV change detection vs first keyframe
        if self.keyframes:
            start_fov = self.keyframes[0].fov
            if pose.fov < start_fov - 3:
                parts.append("zoom in")
            elif pose.fov > start_fov + 3:
                parts.append("zoom out")

        # Movement direction
        if len(self.keyframes) >= 2:
            prev = self.get_pose(max(0, frame - 3))
            dx = pose.position[0] - prev.position[0]
            dy = pose.position[1] - prev.position[1]
            dz = pose.position[2] - prev.position[2]
            if abs(dx) > 0.01:
                parts.append("camera panning right" if dx > 0 else "camera panning left")
            if abs(dy) > 0.01:
                parts.append("camera tilting up" if dy > 0 else "camera tilting down")
            if abs(dz) > 0.01:
                parts.append("dolly backward" if dz > 0 else "dolly forward")

        # Roll
        if abs(pose.rotation[2]) > 5:
            parts.append(f"dutch angle {pose.rotation[2]:.0f}°")

        return ", ".join(parts) if parts else ""

    @classmethod
    def from_preset(cls, preset: str, total_frames: int = 49) -> "CameraPath":
        """
        Named presets for common camera movements.
        """
        mid = total_frames // 2
        presets = {
            "zoom_in": [
                CameraKeyframe(0,           position=(0,0,5.0), fov=60),
                CameraKeyframe(total_frames, position=(0,0,2.0), fov=35),
            ],
            "zoom_out": [
                CameraKeyframe(0,           position=(0,0,2.0), fov=35),
                CameraKeyframe(total_frames, position=(0,0,5.0), fov=60),
            ],
            "pan_left_to_right": [
                CameraKeyframe(0,           position=(-1,0,4), fov=50),
                CameraKeyframe(total_frames, position=( 1,0,4), fov=50),
            ],
            "orbit": [
                CameraKeyframe(0,           position=(0,0,4),  rotation=(0,0,0),   fov=50),
                CameraKeyframe(mid,         position=(4,0,0),  rotation=(0,90,0),  fov=50),
                CameraKeyframe(total_frames, position=(0,0,-4), rotation=(0,180,0), fov=50),
            ],
            "drone_descend": [
                CameraKeyframe(0,           position=(0,5,4),  rotation=(45,0,0), fov=70),
                CameraKeyframe(total_frames, position=(0,1,2),  rotation=(10,0,0), fov=55),
            ],
            "steadicam_forward": [
                CameraKeyframe(0,           position=(0,0,5), fov=50),
                CameraKeyframe(total_frames, position=(0,0,1), fov=50),
            ],
        }
        kfs = presets.get(preset, [CameraKeyframe(0), CameraKeyframe(total_frames)])
        return cls(keyframes=kfs)
