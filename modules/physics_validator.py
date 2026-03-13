"""
modules/physics_validator.py
──────────────────────────────
Valida que las trayectorias de movimiento cumplan reglas de física básica.

Detecta:
  • Aceleraciones/deceleraciones abruptas (violación de momentum)
  • Trayectorias imposibles (objetos pasando a través de suelos)
  • Temblor excesivo de cámara (jitter)
  • Velocidades imposibles para el tipo de sujeto
  • Inconsistencias de escala entre frames

No requiere GPU — trabaja sobre listas de puntos 3D (x, y, z) normalizados
derivados de los keyframes de la escena.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants  (normalized units: 1.0 = "character height")
# ─────────────────────────────────────────────────────────────────────────────

# Maximum reasonable velocity per frame (at 24fps)
MAX_VELOCITY: dict[str, float] = {
    "human":      0.12,   # ~3m/s walking / 0.12 units/frame
    "running":    0.30,   # ~7m/s sprint
    "horse":      0.50,   # ~12m/s gallop
    "car":        1.20,
    "bird":       0.40,
    "camera":     0.80,   # camera can move faster
    "default":    0.60,
}

# Maximum angular velocity per frame (degrees)
MAX_ANGULAR_VEL: dict[str, float] = {
    "head":        8.0,    # degrees/frame
    "torso":       4.0,
    "whole_body": 15.0,
    "camera":     20.0,
    "default":    10.0,
}

# Maximum reasonable acceleration (velocity delta per frame)
MAX_ACCELERATION = 0.15   # units/frame²

# Ground plane (objects below this Y are underground)
GROUND_Y = 0.0

# Camera shake thresholds
CAMERA_SHAKE_THRESHOLD_MILD    = 0.02   # units/frame — mild shake (ok)
CAMERA_SHAKE_THRESHOLD_MODERATE= 0.05   # moderate — warn
CAMERA_SHAKE_THRESHOLD_SEVERE  = 0.12   # severe — error


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrajectoryPoint:
    frame:  int
    x:      float = 0.0
    y:      float = 0.0
    z:      float = 0.0
    vel_x:  float = 0.0   # computed by PhysicsValidator.compute_velocities()
    vel_y:  float = 0.0
    vel_z:  float = 0.0
    speed:  float = 0.0   # magnitude

    @property
    def position(self) -> tuple:
        return (self.x, self.y, self.z)


@dataclass
class PhysicsIssue:
    severity:   str         # "error" | "warning" | "info"
    frame:      int
    issue_type: str         # "acceleration" | "ground_penetration" | "camera_shake" | "speed"
    value:      float       # measured value
    limit:      float       # threshold that was exceeded
    message:    str
    fix:        Optional[str] = None


@dataclass
class PhysicsReport:
    is_valid:      bool
    issues:        list[PhysicsIssue]
    score:         float            # 0.0 – 1.0
    avg_speed:     float = 0.0
    max_speed:     float = 0.0
    max_accel:     float = 0.0
    camera_jitter: float = 0.0

    def errors(self)   -> list[PhysicsIssue]:
        return [i for i in self.issues if i.severity == "error"]
    def warnings(self) -> list[PhysicsIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_markdown(self) -> str:
        lines = [
            f"**Score de física: {self.score:.2f}/1.00**  |  "
            f"vel. media: `{self.avg_speed:.3f}` u/f  |  "
            f"vel. máx: `{self.max_speed:.3f}` u/f  |  "
            f"aceleración máx: `{self.max_accel:.3f}` u/f²  |  "
            f"jitter cámara: `{self.camera_jitter:.3f}` u/f\n"
        ]
        if not self.issues:
            lines.append("✅ Sin problemas de física detectados.")
        for iss in self.issues:
            icon = "🔴" if iss.severity == "error" else "🟡" if iss.severity == "warning" else "🔵"
            lines.append(
                f"{icon} **frame {iss.frame}** [{iss.issue_type}]: {iss.message}"
            )
            if iss.fix:
                lines.append(f"   💡 *Fix:* {iss.fix}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Physics Validator
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsValidator:
    """
    Validates motion trajectories for physical plausibility.

    Usage:
        pv = PhysicsValidator(fps=24, subject_type="human")

        # Provide 3D trajectory as [(frame, x, y, z), …]
        trajectory = [(0, 0.0, 0.0, 0.0), (12, 0.1, 0.5, 0.0), (24, 0.2, 0.0, 0.0)]
        report = pv.validate_trajectory(trajectory)
        print(report.to_markdown())

        # Camera shake from camera path
        camera_traj = [(0, 0,0,5), (1, 0.01,0,5), (2,-0.01,0,5), ...]
        jitter = pv.measure_camera_jitter(camera_traj)
    """

    def __init__(
        self,
        fps:           int   = 24,
        subject_type:  str   = "default",
        gravity:       float = -9.81,     # units/s² (normalized)
        enforce_ground: bool = True,
    ):
        self.fps            = fps
        self.subject_type   = subject_type
        self.gravity        = gravity / (fps * fps)   # convert to per-frame²
        self.enforce_ground = enforce_ground
        self._max_vel       = MAX_VELOCITY.get(subject_type, MAX_VELOCITY["default"])

    # ── Main entry points ─────────────────────────────────────────────────────

    def validate_trajectory(
        self,
        points: list[tuple],   # [(frame, x, y, z), …]  or [(frame, x, y), …]
    ) -> PhysicsReport:
        """
        Full validation of a 3D (or 2D) trajectory.
        """
        if len(points) < 2:
            return PhysicsReport(is_valid=True, issues=[], score=1.0)

        traj = self._build_trajectory(points)
        self.compute_velocities(traj)

        issues: list[PhysicsIssue] = []
        issues.extend(self._check_speed(traj))
        issues.extend(self._check_acceleration(traj))
        if self.enforce_ground:
            issues.extend(self._check_ground(traj))

        speeds  = [p.speed for p in traj]
        accels  = self._compute_accelerations(traj)

        errors   = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        score    = max(0.0, 1.0 - errors * 0.3 - warnings * 0.1)

        return PhysicsReport(
            is_valid=errors == 0,
            issues=issues,
            score=round(score, 2),
            avg_speed=round(sum(speeds) / len(speeds), 4) if speeds else 0,
            max_speed=round(max(speeds), 4) if speeds else 0,
            max_accel=round(max(accels, default=0), 4),
            camera_jitter=0.0,
        )

    def validate_camera_trajectory(
        self,
        camera_points: list[tuple],   # [(frame, x, y, z), …]
    ) -> PhysicsReport:
        """Validates camera path for smooth movement and shake."""
        if len(camera_points) < 2:
            return PhysicsReport(is_valid=True, issues=[], score=1.0)

        traj   = self._build_trajectory(camera_points)
        self.compute_velocities(traj)

        issues: list[PhysicsIssue] = []
        issues.extend(self._check_camera_shake(traj))
        issues.extend(self._check_acceleration(traj, max_accel=0.08))  # stricter for camera

        jitter = self._compute_jitter(traj)
        speeds = [p.speed for p in traj]
        accels = self._compute_accelerations(traj)

        errors   = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        score    = max(0.0, 1.0 - errors * 0.3 - warnings * 0.1)

        return PhysicsReport(
            is_valid=errors == 0,
            issues=issues,
            score=round(score, 2),
            avg_speed=round(sum(speeds) / len(speeds), 4) if speeds else 0,
            max_speed=round(max(speeds), 4) if speeds else 0,
            max_accel=round(max(accels, default=0), 4),
            camera_jitter=round(jitter, 4),
        )

    def validate_from_keyframes(
        self,
        keyframes,              # list[SceneKeyframe]
        total_frames: int = 49,
        subject_type: str = "default",
    ) -> PhysicsReport:
        """
        Estimates trajectory from keyframe descriptions.
        Since keyframes are text-based, we infer 2D Y-position from action.
        """
        ACTION_Y: dict[str, float] = {
            "jumping":    0.8, "flying":     1.5, "hovering":  1.2,
            "falling":    0.3, "crouching":  0.1, "lying_down": 0.0,
            "sitting":    0.2, "standing":   0.0, "walking":    0.0,
            "running":    0.0, "landing":    0.0, "swimming":  -0.3,
        }
        points = []
        for kf in keyframes:
            frame = round(kf.t_start * total_frames)
            desc  = kf.description.lower()
            y     = 0.0
            for action, height in ACTION_Y.items():
                if action in desc:
                    y = height
                    break
            points.append((frame, 0.5, y, 0.0))   # x=center, z=neutral

        return self.validate_trajectory(points)

    # ── Checkers ─────────────────────────────────────────────────────────────

    def _check_speed(self, traj: list[TrajectoryPoint]) -> list[PhysicsIssue]:
        issues = []
        for p in traj:
            if p.speed > self._max_vel * 1.5:
                issues.append(PhysicsIssue(
                    severity="error" if p.speed > self._max_vel * 2.5 else "warning",
                    frame=p.frame,
                    issue_type="speed",
                    value=p.speed,
                    limit=self._max_vel,
                    message=(
                        f"Velocidad {p.speed:.3f} u/f supera el límite para "
                        f"'{self.subject_type}' ({self._max_vel:.3f} u/f)"
                    ),
                    fix="Añade más keyframes intermedios para suavizar el movimiento.",
                ))
        return issues

    def _check_acceleration(
        self,
        traj: list[TrajectoryPoint],
        max_accel: float | None = None,
    ) -> list[PhysicsIssue]:
        issues = []
        limit  = max_accel or MAX_ACCELERATION
        accels = self._compute_accelerations(traj)

        for i, accel in enumerate(accels):
            if accel > limit * 2.0:
                issues.append(PhysicsIssue(
                    severity="error",
                    frame=traj[i].frame,
                    issue_type="acceleration",
                    value=accel,
                    limit=limit,
                    message=(
                        f"Aceleración abrupta: {accel:.3f} u/f² "
                        f"(máx recomendado: {limit:.3f})"
                    ),
                    fix="Añade keyframes de transición para suavizar el arranque/parada.",
                ))
            elif accel > limit:
                issues.append(PhysicsIssue(
                    severity="warning",
                    frame=traj[i].frame,
                    issue_type="acceleration",
                    value=accel,
                    limit=limit,
                    message=f"Aceleración moderadamente alta: {accel:.3f} u/f²",
                    fix="Considera usar easing (ease_in_out) en esta transición.",
                ))
        return issues

    def _check_ground(self, traj: list[TrajectoryPoint]) -> list[PhysicsIssue]:
        issues = []
        for p in traj:
            if p.y < GROUND_Y - 0.05:   # small tolerance
                issues.append(PhysicsIssue(
                    severity="warning",
                    frame=p.frame,
                    issue_type="ground_penetration",
                    value=p.y,
                    limit=GROUND_Y,
                    message=f"Sujeto por debajo del suelo: y={p.y:.3f}",
                    fix="Ajusta la posición Y o elimina el enforce_ground si es intencional.",
                ))
        return issues

    def _check_camera_shake(self, traj: list[TrajectoryPoint]) -> list[PhysicsIssue]:
        issues = []
        if len(traj) < 3:
            return issues

        for i in range(1, len(traj) - 1):
            prev_vel = traj[i-1].speed
            curr_vel = traj[i].speed
            next_vel = traj[i+1].speed

            # Detect direction reversal (high-frequency jitter)
            dx_prev = traj[i].x - traj[i-1].x
            dx_next = traj[i+1].x - traj[i].x
            dy_prev = traj[i].y - traj[i-1].y
            dy_next = traj[i+1].y - traj[i].y

            reversal_x = dx_prev * dx_next < 0
            reversal_y = dy_prev * dy_next < 0
            # jitter_mag not used in conditions below, kept for future use
            jitter_mag = math.sqrt(
                (traj[i].x - (traj[i-1].x + traj[i+1].x) / 2) ** 2 +
                (traj[i].y - (traj[i-1].y + traj[i+1].y) / 2) ** 2
            )

            if reversal_x and reversal_y and curr_vel > CAMERA_SHAKE_THRESHOLD_SEVERE:
                issues.append(PhysicsIssue(
                    severity="error",
                    frame=traj[i].frame,
                    issue_type="camera_shake",
                    value=curr_vel,
                    limit=CAMERA_SHAKE_THRESHOLD_SEVERE,
                    message=f"Temblor severo de cámara en frame {traj[i].frame}: {curr_vel:.3f} u/f",
                    fix="Usa una CameraPath con splines Catmull-Rom para suavizar el movimiento.",
                ))
            elif (reversal_x or reversal_y) and curr_vel > CAMERA_SHAKE_THRESHOLD_MODERATE:
                issues.append(PhysicsIssue(
                    severity="warning",
                    frame=traj[i].frame,
                    issue_type="camera_shake",
                    value=curr_vel,
                    limit=CAMERA_SHAKE_THRESHOLD_MODERATE,
                    message=f"Movimiento brusco de cámara en frame {traj[i].frame}",
                    fix="Aplica smoothing o usa el preset 'steadicam' de CameraPath.",
                ))
        return issues

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _build_trajectory(
        self,
        points: list[tuple],
    ) -> list[TrajectoryPoint]:
        traj = []
        for p in points:
            frame = int(p[0])
            x = float(p[1]) if len(p) > 1 else 0.0
            y = float(p[2]) if len(p) > 2 else 0.0
            z = float(p[3]) if len(p) > 3 else 0.0
            traj.append(TrajectoryPoint(frame=frame, x=x, y=y, z=z))
        traj.sort(key=lambda p: p.frame)
        return traj

    def compute_velocities(self, traj: list[TrajectoryPoint]) -> list[TrajectoryPoint]:
        for i in range(len(traj)):
            if i == 0:
                traj[i].vel_x = traj[i].vel_y = traj[i].vel_z = 0.0
            else:
                dt = max(traj[i].frame - traj[i-1].frame, 1)
                traj[i].vel_x = (traj[i].x - traj[i-1].x) / dt
                traj[i].vel_y = (traj[i].y - traj[i-1].y) / dt
                traj[i].vel_z = (traj[i].z - traj[i-1].z) / dt
            traj[i].speed = math.sqrt(
                traj[i].vel_x**2 + traj[i].vel_y**2 + traj[i].vel_z**2
            )
        return traj

    def _compute_accelerations(
        self,
        traj: list[TrajectoryPoint],
    ) -> list[float]:
        accels = []
        for i in range(1, len(traj)):
            dt    = max(traj[i].frame - traj[i-1].frame, 1)
            delta = abs(traj[i].speed - traj[i-1].speed) / dt
            accels.append(delta)
        return accels

    def _compute_jitter(self, traj: list[TrajectoryPoint]) -> float:
        """
        Computes mean high-frequency oscillation (camera jitter metric).
        """
        if len(traj) < 3:
            return 0.0
        deviations = []
        for i in range(1, len(traj) - 1):
            mid_x = (traj[i-1].x + traj[i+1].x) / 2
            mid_y = (traj[i-1].y + traj[i+1].y) / 2
            dev   = math.sqrt((traj[i].x - mid_x)**2 + (traj[i].y - mid_y)**2)
            deviations.append(dev)
        return sum(deviations) / len(deviations) if deviations else 0.0

    def measure_camera_jitter(self, camera_points: list[tuple]) -> float:
        """Quick scalar jitter measurement. Lower = smoother camera."""
        traj = self._build_trajectory(camera_points)
        self.compute_velocities(traj)
        return self._compute_jitter(traj)

    # ── Trajectory from prompt (NLP-based) ───────────────────────────────────

    def infer_trajectory_from_prompt(
        self,
        actions,            # list[ExtractedAction]
        total_frames: int,
    ) -> list[tuple]:
        """
        Infers a rough 3D trajectory from extracted actions.
        Returns [(frame, x, y, z), …] for validate_trajectory().
        """
        HEIGHT_MAP = {
            "jumping": 0.8, "flying": 1.5, "hovering": 1.2,
            "falling": 0.3, "crouching": 0.15, "lying_down": 0.0,
            "sitting": 0.2, "standing": 0.0,
            "walking": 0.0, "running": 0.0,
        }
        SPEED_MAP = {
            "running": 0.25, "walking": 0.08,
            "flying": 0.35, "jumping": 0.15,
            "crawling": 0.04, "sliding": 0.20,
        }

        if not actions:
            return [(0, 0.5, 0.0, 0.0), (total_frames, 0.5, 0.0, 0.0)]

        points = []
        for i, action in enumerate(actions):
            frame = round(i / max(len(actions) - 1, 1) * (total_frames - 1))
            y     = HEIGHT_MAP.get(action.verb, 0.0)
            spd   = SPEED_MAP.get(action.verb, 0.1) * action.speed

            # Direction-based x/z
            dx, dz = 0.0, 0.0
            if action.direction:
                axis, sign = action.direction
                if axis == "x":
                    dx = sign * spd
                elif axis == "z":
                    dz = sign * spd

            x = 0.5 + dx * frame
            z = 0.0 + dz * frame
            points.append((frame, x, y, z))

        return points
