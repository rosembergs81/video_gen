"""
modules/spatial_control.py
────────────────────────────
Renderiza mapas de control espacial (Pose/Depth simplificados) a partir
del SceneGraph y SkeletalAnimator.

Funcionalidad:
  • Genera un "depth map" simplificado en función de las posiciones 3D
    del SceneGraph (objetos más cercanos = más brillantes).
  • Genera un "pose overlay" tipo OpenPose a partir de las articulaciones
    del SkeletalAnimator interpoladas para un frame dado.
  • Combina ambos en una imagen que se puede inyectar como primer frame
    para guiar un pipeline I2V (CogVideoX-5B-I2V), forzando coherencia
    espacial sin depender únicamente del entendimiento semántico del LLM.

Esto funciona como un "poor-man's ControlNet": no inyecta condiciones
latentes, pero ancla visualmente la composición del primer frame.
"""

from __future__ import annotations
import math
from PIL import Image, ImageDraw, ImageFont
from typing import Optional

# ── OpenPose-like color palette for joints/limbs ─────────────────────────────
JOINT_COLORS = {
    "head":           (255, 0, 0),       # Red
    "neck":           (255, 85, 0),
    "left_shoulder":  (255, 170, 0),
    "right_shoulder": (0, 255, 0),
    "left_elbow":     (0, 255, 85),
    "right_elbow":    (0, 255, 170),
    "left_wrist":     (0, 170, 255),
    "right_wrist":    (0, 85, 255),
    "torso":          (255, 255, 0),
    "left_hip":       (170, 255, 0),
    "right_hip":      (85, 255, 0),
    "left_knee":      (0, 255, 255),
    "right_knee":     (255, 0, 255),
    "left_ankle":     (255, 0, 170),
    "right_ankle":    (255, 0, 85),
    # Quadruped / Bird
    "front_left_leg":  (0, 255, 170),
    "front_right_leg": (0, 255, 85),
    "hind_left_leg":   (0, 170, 255),
    "hind_right_leg":  (0, 85, 255),
    "tail":            (170, 0, 255),
    "left_wing":       (255, 170, 0),
    "right_wing":      (0, 255, 170),
    "body":            (255, 255, 0),
    "spine":           (255, 200, 0),
}

# Limb connections for human skeleton (OpenPose-style)
HUMAN_LIMBS = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("neck", "torso"),
    ("torso", "left_hip"),
    ("torso", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

QUADRUPED_LIMBS = [
    ("head", "neck"),
    ("neck", "spine"),
    ("spine", "body"),
    ("body", "tail"),
    ("neck", "front_left_leg"),
    ("neck", "front_right_leg"),
    ("body", "hind_left_leg"),
    ("body", "hind_right_leg"),
]

BIRD_LIMBS = [
    ("head", "neck"),
    ("neck", "body"),
    ("body", "tail"),
    ("neck", "left_wing"),
    ("neck", "right_wing"),
    ("body", "left_ankle"),
    ("body", "right_ankle"),
]

SKELETON_LIMBS = {
    "human": HUMAN_LIMBS,
    "quadruped": QUADRUPED_LIMBS,
    "bird": BIRD_LIMBS,
}


def _joint_to_pixel(joint_name: str, pose_dict: dict, width: int, height: int,
                    offset_x: float = 0.0, offset_y: float = 0.0) -> tuple[int, int] | None:
    """
    Convert joint position data to pixel coordinates.
    If px/py keys exist use those; otherwise estimate from angles.
    """
    px_key = f"{joint_name}_px"
    py_key = f"{joint_name}_py"

    if px_key in pose_dict and py_key in pose_dict:
        x = (pose_dict[px_key] + offset_x) * width
        y = (pose_dict[py_key] + offset_y) * height
        return (int(x), int(y))

    # Fallback: use a basic anatomical layout from angles
    # This gives approximate positions for visualization
    ax_key = f"{joint_name}_ax"
    if ax_key not in pose_dict:
        return None

    # Default positions for human joints (normalized)
    DEFAULT_POS = {
        "head":           (0.50, 0.10),
        "neck":           (0.50, 0.18),
        "left_shoulder":  (0.38, 0.22),
        "right_shoulder": (0.62, 0.22),
        "left_elbow":     (0.30, 0.38),
        "right_elbow":    (0.70, 0.38),
        "left_wrist":     (0.25, 0.52),
        "right_wrist":    (0.75, 0.52),
        "torso":          (0.50, 0.42),
        "left_hip":       (0.42, 0.55),
        "right_hip":      (0.58, 0.55),
        "left_knee":      (0.40, 0.72),
        "right_knee":     (0.60, 0.72),
        "left_ankle":     (0.38, 0.90),
        "right_ankle":    (0.62, 0.90),
        # Quadruped
        "spine":          (0.50, 0.35),
        "body":           (0.55, 0.45),
        "tail":           (0.80, 0.40),
        "front_left_leg": (0.35, 0.75),
        "front_right_leg":(0.45, 0.75),
        "hind_left_leg":  (0.65, 0.75),
        "hind_right_leg": (0.75, 0.75),
        # Bird
        "left_wing":      (0.25, 0.30),
        "right_wing":     (0.75, 0.30),
    }

    base = DEFAULT_POS.get(joint_name, (0.5, 0.5))
    # Apply small angle-based perturbation
    angle = pose_dict.get(ax_key, 0.0)
    dx = math.sin(math.radians(angle)) * 0.05
    dy = math.cos(math.radians(angle)) * 0.02

    x = (base[0] + dx + offset_x) * width
    y = (base[1] + dy + offset_y) * height
    return (int(x), int(y))


def render_pose_overlay(
    pose_dict: dict,
    skeleton_type: str = "human",
    width: int = 720,
    height: int = 480,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    bg_image: Image.Image | None = None,
) -> Image.Image:
    """
    Renders an OpenPose-style skeleton overlay from pose joint data.
    
    Args:
        pose_dict: Flat dict of joint values (from SkeletalAnimator)
        skeleton_type: 'human', 'quadruped', or 'bird'
        width, height: Output image dimensions
        offset_x, offset_y: Translate the skeleton position (0-1 normalized)
        bg_image: Optional background image; if None, uses black
    
    Returns:
        PIL Image with skeleton overlay
    """
    if bg_image is not None:
        img = bg_image.copy().resize((width, height), Image.LANCZOS)
    else:
        img = Image.new("RGB", (width, height), (0, 0, 0))

    draw = ImageDraw.Draw(img)
    limbs = SKELETON_LIMBS.get(skeleton_type, HUMAN_LIMBS)

    # Draw limb connections
    for joint_a, joint_b in limbs:
        pos_a = _joint_to_pixel(joint_a, pose_dict, width, height, offset_x, offset_y)
        pos_b = _joint_to_pixel(joint_b, pose_dict, width, height, offset_x, offset_y)
        if pos_a and pos_b:
            color = JOINT_COLORS.get(joint_a, (200, 200, 200))
            draw.line([pos_a, pos_b], fill=color, width=4)

    # Draw joint circles
    all_joints = set()
    for a, b in limbs:
        all_joints.add(a)
        all_joints.add(b)

    for joint_name in all_joints:
        pos = _joint_to_pixel(joint_name, pose_dict, width, height, offset_x, offset_y)
        if pos:
            color = JOINT_COLORS.get(joint_name, (255, 255, 255))
            r = 6
            draw.ellipse([pos[0]-r, pos[1]-r, pos[0]+r, pos[1]+r], fill=color)

    return img


def render_depth_map(
    scene_graph,
    width: int = 720,
    height: int = 480,
    frame: int = 0,
) -> Image.Image:
    """
    Renders a simplified depth map from the SceneGraph.
    Objects closer to the camera (lower z) appear brighter.
    
    This gives the I2V pipeline spatial cues about subject positions
    and relative depths.
    """
    img = Image.new("L", (width, height), 0)  # Grayscale black
    draw = ImageDraw.Draw(img)

    if not hasattr(scene_graph, "objects") or not scene_graph.objects:
        return img.convert("RGB")

    for obj in scene_graph.objects.values():
        if not obj.visible:
            continue

        x = obj.position[0] * width
        y = obj.position[1] * height
        z = obj.position[2] if len(obj.position) > 2 else 0.0

        # Brightness inversely proportional to z (closer = brighter)
        brightness = int(max(50, min(255, 255 * (1.0 - (z + 0.5)))))
        
        # Size proportional to object size, inversely to z
        base_r = int(40 * obj.size * max(0.3, 1.0 - z * 0.5))

        # Draw blob with gradient
        for layer in range(base_r, 0, -2):
            layer_brightness = int(brightness * (layer / base_r))
            draw.ellipse(
                [x - layer, y - layer, x + layer, y + layer],
                fill=layer_brightness
            )

        # Add label
        try:
            draw.text((int(x) - 20, int(y) + base_r + 5),
                      obj.label, fill=brightness)
        except Exception:
            pass

    return img.convert("RGB")


def render_spatial_control_image(
    scene_graph,
    skeleton_type: str = "human",
    pose_dict: dict | None = None,
    width: int = 720,
    height: int = 480,
    frame: int = 0,
    blend_alpha: float = 0.6,
) -> Image.Image:
    """
    Composite render: depth map + pose overlay.
    
    This produces a single image that encodes:
      1. Spatial positioning of objects (depth map)
      2. Skeleton pose of the primary character (OpenPose overlay)
    
    The result can be injected as the `image` input for CogVideoX-I2V,
    effectively acting as a "poor-man's ControlNet" that anchors
    the spatial layout of the generated video.
    
    Args:
        scene_graph: SceneGraph instance with object positions
        skeleton_type: Type of skeleton ('human', 'quadruped', 'bird')
        pose_dict: Joint data from SkeletalAnimator (flat dict)
        width, height: Output image dimensions
        frame: Current frame number for depth rendering
        blend_alpha: How much the pose overlay dominates vs depth map (0-1)
    
    Returns:
        PIL Image ready for I2V injection
    """
    import numpy as np

    # Render depth map as base
    depth_img = render_depth_map(scene_graph, width, height, frame)

    if pose_dict is None:
        return depth_img

    # Find offset based on primary character position in scene graph
    offset_x, offset_y = 0.0, 0.0
    if hasattr(scene_graph, "objects") and scene_graph.objects:
        first_obj = next(iter(scene_graph.objects.values()))
        offset_x = first_obj.position[0] - 0.5
        offset_y = first_obj.position[1] - 0.5

    # Render pose overlay on transparent/black background
    pose_img = render_pose_overlay(
        pose_dict, skeleton_type, width, height,
        offset_x=offset_x, offset_y=offset_y
    )

    # Alpha blend
    depth_arr = np.array(depth_img).astype(np.float32)
    pose_arr = np.array(pose_img).astype(np.float32)

    # Where pose has content, blend; otherwise use depth
    pose_mask = np.max(pose_arr, axis=-1, keepdims=True) > 10
    blended = np.where(
        pose_mask,
        depth_arr * (1 - blend_alpha) + pose_arr * blend_alpha,
        depth_arr
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)
