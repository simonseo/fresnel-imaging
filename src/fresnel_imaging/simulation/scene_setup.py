"""Scene configuration for Fresnel lens optical simulation.

Sets up calibration (checkerboard) and imaging scenes with correct
object / sensor placement using the thin lens equation.
"""

from __future__ import annotations

import math
import os
from typing import Any

try:
    import bmesh
    import bpy

    _HAS_BPY = True
except ImportError:
    bpy = None  # type: ignore[assignment]
    bmesh = None  # type: ignore[assignment]
    _HAS_BPY = False


def _require_bpy() -> None:
    if not _HAS_BPY:
        raise RuntimeError(
            "This module must be run inside Blender. "
            "Use: blender -b -P <script.py>"
        )


def _focal_length_from_obj(lens_obj: Any) -> float:
    return float(lens_obj.get("fresnel_focal_length", 200.0))


def _thin_lens_image_distance(focal_length: float, object_distance: float) -> float:
    """Compute image distance from the thin lens equation: 1/f = 1/do + 1/di."""
    if object_distance <= focal_length:
        raise ValueError(
            f"Object distance ({object_distance}) must be greater than "
            f"focal length ({focal_length}) for a real image."
        )
    return 1.0 / (1.0 / focal_length - 1.0 / object_distance)


def create_checkerboard_pattern(
    rows: int = 8,
    cols: int = 8,
    square_size: float = 25.0,
    name: str = "Checkerboard",
) -> Any:
    """Generate a checkerboard plane mesh with alternating B&W materials.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 8×8).
    square_size : float
        Size of each square in mm.
    name : str
        Blender object name.

    Returns
    -------
    bpy.types.Object
    """
    _require_bpy()
    from .materials import create_diffuse_material

    bm = bmesh.new()

    for row in range(rows):
        for col in range(cols):
            x0 = (col - cols / 2.0) * square_size
            y0 = (row - rows / 2.0) * square_size
            x1 = x0 + square_size
            y1 = y0 + square_size

            v0 = bm.verts.new((x0, y0, 0.0))
            v1 = bm.verts.new((x1, y0, 0.0))
            v2 = bm.verts.new((x1, y1, 0.0))
            v3 = bm.verts.new((x0, y1, 0.0))
            face = bm.faces.new((v0, v1, v2, v3))
            # Material index: 0 = white, 1 = black (alternating)
            face.material_index = (row + col) % 2

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    white_mat = create_diffuse_material(f"{name}_White", color=(0.9, 0.9, 0.9))
    black_mat = create_diffuse_material(f"{name}_Black", color=(0.02, 0.02, 0.02))
    obj.data.materials.append(white_mat)
    obj.data.materials.append(black_mat)

    return obj


def _create_camera(
    location: tuple[float, float, float],
    look_at: tuple[float, float, float],
    name: str = "SimCamera",
    sensor_width: float = 36.0,
) -> Any:
    """Create a camera at *location* pointing toward *look_at*."""
    _require_bpy()

    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_width = sensor_width
    cam_data.type = "PERSP"

    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = location

    # Point camera along the optical axis (negative Z toward target)
    direction = (
        look_at[0] - location[0],
        look_at[1] - location[1],
        look_at[2] - location[2],
    )
    dist = math.sqrt(sum(d * d for d in direction))
    if dist > 1e-9:
        # Camera default looks down -Z; we need to rotate it
        # Use track-to constraint for reliable aiming
        constraint = cam_obj.constraints.new(type="TRACK_TO")
        # Create empty target for the constraint
        target = bpy.data.objects.new(f"{name}_Target", None)
        target.location = look_at
        bpy.context.collection.objects.link(target)
        constraint.target = target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"

    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_calibration_scene(
    lens_obj: Any,
    pattern_distance: float = 500.0,
    sensor_distance: float | None = None,
    rows: int = 8,
    cols: int = 8,
    square_size: float = 25.0,
) -> dict[str, Any]:
    """Set up a calibration scene with a checkerboard target.

    Places a checkerboard pattern at ``pattern_distance`` mm in front of
    the lens (along −Z) and positions a camera/sensor at the image plane
    behind the lens (along +Z), computed from the thin lens equation.

    The lens object is assumed centred at the origin with its optical
    axis along Z.

    Parameters
    ----------
    lens_obj : bpy.types.Object
        The Fresnel lens object (created by :func:`create_fresnel_lens`).
    pattern_distance : float
        Distance from lens to checkerboard in mm (default 500).
    sensor_distance : float | None
        Override image distance. If *None*, computed from thin lens eq.
    rows, cols : int
        Checkerboard grid size.
    square_size : float
        Size of each checkerboard square in mm.

    Returns
    -------
    dict
        ``{"checkerboard": obj, "camera": obj, "image_distance": float}``
    """
    _require_bpy()

    focal_length = _focal_length_from_obj(lens_obj)

    if sensor_distance is None:
        sensor_distance = _thin_lens_image_distance(focal_length, pattern_distance)

    checkerboard = create_checkerboard_pattern(rows, cols, square_size)
    # Place checkerboard in front of the lens (−Z direction)
    checkerboard.location = (0.0, 0.0, -pattern_distance)
    # Rotate to face the lens (pattern normal along +Z)
    checkerboard.rotation_euler = (0.0, 0.0, 0.0)

    # Place camera behind the lens at the image plane (+Z direction)
    camera = _create_camera(
        location=(0.0, 0.0, sensor_distance),
        look_at=(0.0, 0.0, 0.0),
    )

    # Compute FoV to frame the checkerboard
    board_half = max(rows, cols) * square_size / 2.0
    magnification = sensor_distance / pattern_distance
    image_half = board_half * magnification
    cam_data = camera.data
    cam_data.lens = (cam_data.sensor_width / 2.0) / math.tan(
        math.atan2(image_half, sensor_distance)
    )

    return {
        "checkerboard": checkerboard,
        "camera": camera,
        "image_distance": sensor_distance,
    }


def setup_imaging_scene(
    lens_obj: Any,
    image_path: str,
    distance: float = 1000.0,
    sensor_distance: float | None = None,
    plane_size: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Place an image-textured plane as the scene subject.

    Parameters
    ----------
    lens_obj : bpy.types.Object
        The Fresnel lens object.
    image_path : str
        Path to the image file to use as texture.
    distance : float
        Distance from lens to image plane in mm.
    sensor_distance : float | None
        Override image distance. If *None*, computed from thin lens eq.
    plane_size : tuple[float, float] | None
        ``(width, height)`` of the plane in mm. If *None*, derived from
        the image aspect ratio scaled to 200 mm width.

    Returns
    -------
    dict
        ``{"image_plane": obj, "camera": obj, "image_distance": float}``
    """
    _require_bpy()

    focal_length = _focal_length_from_obj(lens_obj)

    if sensor_distance is None:
        sensor_distance = _thin_lens_image_distance(focal_length, distance)

    # Load image
    img = bpy.data.images.load(os.path.abspath(image_path))
    w_px, h_px = img.size

    if plane_size is None:
        width = 200.0
        height = width * (h_px / w_px) if w_px > 0 else width
    else:
        width, height = plane_size

    # Create textured plane
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, -distance))
    plane = bpy.context.active_object
    plane.name = "ImagePlane"
    plane.scale = (width, height, 1.0)
    bpy.ops.object.transform_apply(scale=True)

    # Create image texture material
    mat = bpy.data.materials.new("ImageTexture")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output_node = tree.nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (400, 0)

    # Use emission so the image is self-lit (acts as a light source)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.location = (200, 0)
    emission.inputs["Strength"].default_value = 5.0

    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.location = (0, 0)
    tex.image = img

    tree.links.new(tex.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output_node.inputs["Surface"])

    plane.data.materials.append(mat)

    camera = _create_camera(
        location=(0.0, 0.0, sensor_distance),
        look_at=(0.0, 0.0, 0.0),
    )

    return {
        "image_plane": plane,
        "camera": camera,
        "image_distance": sensor_distance,
    }
