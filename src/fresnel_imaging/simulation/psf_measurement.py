"""PSF measurement automation via point source rendering.

Measures the point spread function (PSF) of a Fresnel lens by rendering
a small emissive sphere (point source) through the lens and capturing
the resulting light distribution on a sensor plane.  An emissive UV
sphere is used instead of a Blender point light because it interacts
correctly with LuxCore glass materials and produces physically accurate
caustic patterns.

Workflow
--------
1. :func:`setup_psf_scene` places a point source, Fresnel lens, and
   camera/sensor along the optical axis, computing the sensor distance
   from the thin lens equation.
2. The scene is rendered externally (``render.render_scene``).
3. :func:`extract_psf_from_render` loads the rendered image and extracts
   a normalised PSF kernel centred on the intensity centroid.
4. For spatially-varying PSFs, :func:`setup_psf_grid_scene` provides
   grid positions; the user renders each point source sequentially.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

try:
    import bmesh
    import bpy

    _HAS_BPY = True
except ImportError:
    bpy = None  # type: ignore[assignment]
    bmesh = None  # type: ignore[assignment]
    _HAS_BPY = False

logger = logging.getLogger(__name__)


def _require_bpy() -> None:
    if not _HAS_BPY:
        raise RuntimeError(
            "This module must be run inside Blender. "
            "Use: blender -b -P <script.py>"
        )


# ------------------------------------------------------------------
# Point source creation
# ------------------------------------------------------------------


def create_point_source(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.1,
    emission_strength: float = 10000.0,
    name: str = "PointSource",
) -> Any:
    """Create a small emissive UV sphere to act as a point source.

    Uses an emission shader rather than a Blender point light so that
    the light interacts correctly with LuxCore glass geometry.

    Parameters
    ----------
    position : tuple[float, float, float]
        World-space location of the sphere centre.
    radius : float
        Sphere radius in scene units (default 0.1 — small enough to
        approximate a point source).
    emission_strength : float
        Emission shader strength (default 10 000).
    name : str
        Blender object name.

    Returns
    -------
    bpy.types.Object
        The emissive sphere object.
    """
    _require_bpy()

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=8,
        ring_count=4,
        radius=radius,
        location=position,
    )
    sphere = bpy.context.active_object
    sphere.name = name

    # --- emissive material ---
    mat = bpy.data.materials.new(f"{name}_Emission")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output_node = tree.nodes.new("ShaderNodeOutputMaterial")
    output_node.location = (300, 0)

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.location = (0, 0)
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = emission_strength

    tree.links.new(
        emission.outputs["Emission"],
        output_node.inputs["Surface"],
    )

    sphere.data.materials.clear()
    sphere.data.materials.append(mat)

    logger.info(
        "Created point source '%s' at %s (r=%.3f, strength=%.0f)",
        name,
        position,
        radius,
        emission_strength,
    )
    return sphere


# ------------------------------------------------------------------
# Sensor plane
# ------------------------------------------------------------------


def create_sensor_plane(
    location: tuple[float, float, float],
    size: float = 36.0,
    name: str = "SensorPlane",
) -> Any:
    """Create a matte-white plane to act as a virtual sensor.

    Parameters
    ----------
    location : tuple[float, float, float]
        World-space position for the plane centre.
    size : float
        Side length of the square plane (default 36 mm — full-frame).
    name : str
        Blender object name.

    Returns
    -------
    bpy.types.Object
        The sensor plane object.
    """
    _require_bpy()
    from .materials import create_diffuse_material

    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    plane = bpy.context.active_object
    plane.name = name

    white_mat = create_diffuse_material(
        f"{name}_White", color=(0.9, 0.9, 0.9)
    )
    plane.data.materials.clear()
    plane.data.materials.append(white_mat)

    logger.info(
        "Created sensor plane '%s' at %s (size=%.1f)",
        name,
        location,
        size,
    )
    return plane


# ------------------------------------------------------------------
# PSF scene assembly
# ------------------------------------------------------------------


def setup_psf_scene(
    lens_obj: Any,
    point_position: tuple[float, float, float] = (0.0, 0.0, -500.0),
    sensor_distance: float | None = None,
    render_samples: int = 5000,
    render_resolution: tuple[int, int] = (512, 512),
) -> dict[str, Any]:
    """Assemble a scene for measuring a single on/off-axis PSF.

    Places a point source at *point_position* (in front of the lens
    along −Z) and a camera at the image plane behind the lens (+Z),
    with the sensor distance computed from the thin lens equation.

    Parameters
    ----------
    lens_obj : bpy.types.Object
        Fresnel lens object (must have ``fresnel_focal_length`` or
        ``lens_focal_length`` custom property).
    point_position : tuple[float, float, float]
        Location of the point source (default on-axis at z = −500).
    sensor_distance : float | None
        Override image-plane distance.  If *None*, computed via the
        thin lens equation.
    render_samples : int
        Number of render samples (default 5 000).
    render_resolution : tuple[int, int]
        ``(width, height)`` in pixels.

    Returns
    -------
    dict
        ``{"point_source", "camera", "sensor_distance",
        "render_config"}``
    """
    _require_bpy()
    from .scene_setup import _create_camera, _thin_lens_image_distance

    # Focal length from lens custom properties
    focal_length = float(
        lens_obj.get(
            "fresnel_focal_length",
            lens_obj.get("lens_focal_length", 200.0),
        )
    )

    # Compute sensor distance from thin-lens equation if not given
    if sensor_distance is None:
        object_distance = abs(point_position[2])
        sensor_distance = _thin_lens_image_distance(
            focal_length, object_distance
        )

    # Create point source
    point_source = create_point_source(position=point_position)

    # Create camera at sensor plane looking back at origin
    camera = _create_camera(
        location=(0.0, 0.0, sensor_distance),
        look_at=(0.0, 0.0, 0.0),
    )

    # ---- Render configuration ----
    scene = bpy.context.scene

    # Try LuxCore first (best for glass caustics)
    luxcore_configured = False
    try:
        from .render import configure_luxcore_render

        configure_luxcore_render(
            samples=render_samples,
            engine="BIDIR",
            resolution=render_resolution,
            light_depth=25,
            path_depth=25,
        )
        luxcore_configured = True
        logger.info("Configured LuxCore render (BIDIR, %d spp)", render_samples)
    except Exception:
        logger.info("LuxCore not available; falling back to Cycles")

    if not luxcore_configured:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = render_samples
        scene.cycles.max_bounces = 32
        scene.cycles.glossy_bounces = 16
        scene.cycles.transmission_bounces = 25
        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True

        scene.render.resolution_x = render_resolution[0]
        scene.render.resolution_y = render_resolution[1]
        scene.render.resolution_percentage = 100

    # EXR output for full dynamic range
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"

    render_config: dict[str, Any] = {
        "samples": render_samples,
        "resolution": render_resolution,
        "engine": "LUXCORE" if luxcore_configured else "CYCLES",
        "output_format": "OPEN_EXR",
    }

    logger.info(
        "PSF scene ready: sensor_distance=%.2f, engine=%s",
        sensor_distance,
        render_config["engine"],
    )

    return {
        "point_source": point_source,
        "camera": camera,
        "sensor_distance": sensor_distance,
        "render_config": render_config,
    }


# ------------------------------------------------------------------
# PSF extraction (pure NumPy — no Blender dependency)
# ------------------------------------------------------------------


def extract_psf_from_render(
    source: str | pathlib.Path | np.ndarray,
    psf_size: int = 31,
    threshold: float = 0.001,
) -> np.ndarray:
    """Extract a normalised PSF kernel from a rendered image.

    This is a **pure NumPy** function with no Blender dependency.

    Parameters
    ----------
    source : str | pathlib.Path | np.ndarray
        Path to a rendered image (EXR/PNG/TIFF, loaded via OpenCV) or
        an already-loaded array (H×W or H×W×C).
    psf_size : int
        Side length of the square PSF kernel (default 31, must be odd
        for a centred kernel — even values are accepted but the centre
        pixel will be slightly off).
    threshold : float
        Minimum total intensity.  If the image is darker than this a
        uniform PSF is returned instead of a degenerate centroid.

    Returns
    -------
    np.ndarray
        Shape ``(psf_size, psf_size)``, dtype ``float64``, sum ≈ 1.0.

    Notes
    -----
    When the source is loaded via OpenCV the channel order is BGR;
    when a user passes an ndarray directly it may be RGB.  The
    luminance weights ``0.2126·R + 0.7152·G + 0.0722·B`` are applied
    in the array's native channel order, which gives slightly different
    (but acceptable) results for BGR input.  For exact correctness,
    convert to grayscale before calling.
    """
    # ---- Load image ----
    if isinstance(source, (str, pathlib.Path)):
        import cv2

        img = cv2.imread(
            str(source),
            cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH,
        )
        if img is None:
            raise FileNotFoundError(
                f"Could not load image: {source}"
            )
        img = img.astype(np.float64)
    else:
        img = np.asarray(source, dtype=np.float64)

    # ---- Convert to grayscale ----
    if img.ndim == 3:
        if img.shape[2] >= 3:
            img = (
                0.2126 * img[:, :, 0]
                + 0.7152 * img[:, :, 1]
                + 0.0722 * img[:, :, 2]
            )
        else:
            img = img[:, :, 0]

    # ---- Find intensity centroid ----
    total = float(img.sum())
    if total < threshold:
        logger.warning(
            "Total intensity (%.6f) below threshold; "
            "returning uniform PSF",
            total,
        )
        return np.full(
            (psf_size, psf_size), 1.0 / (psf_size * psf_size),
            dtype=np.float64,
        )

    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = float(np.sum(yy * img) / total)
    cx = float(np.sum(xx * img) / total)

    cy_int = int(round(cy))
    cx_int = int(round(cx))

    # ---- Crop around centroid ----
    half = psf_size // 2
    r0 = cy_int - half
    r1 = r0 + psf_size
    c0 = cx_int - half
    c1 = c0 + psf_size

    # If centroid is near edge, create zero-padded output
    psf = np.zeros((psf_size, psf_size), dtype=np.float64)

    # Compute overlap between source image and crop window
    src_r0 = max(r0, 0)
    src_r1 = min(r1, h)
    src_c0 = max(c0, 0)
    src_c1 = min(c1, w)

    dst_r0 = src_r0 - r0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c0 = src_c0 - c0
    dst_c1 = dst_c0 + (src_c1 - src_c0)

    psf[dst_r0:dst_r1, dst_c0:dst_c1] = img[
        src_r0:src_r1, src_c0:src_c1
    ]

    # ---- Clip negatives and normalise ----
    np.maximum(psf, 0.0, out=psf)
    psf_sum = float(psf.sum())
    if psf_sum > 0.0:
        psf /= psf_sum
    else:
        psf[:] = 1.0 / (psf_size * psf_size)

    return psf


# ------------------------------------------------------------------
# Spatially-varying PSF grid
# ------------------------------------------------------------------


def setup_psf_grid_scene(
    lens_obj: Any,
    grid_rows: int = 5,
    grid_cols: int = 5,
    grid_spacing: float = 20.0,
    object_distance: float = 500.0,
    render_samples: int = 5000,
    render_resolution: tuple[int, int] = (512, 512),
) -> dict[str, Any]:
    """Prepare grid positions for spatially-varying PSF measurement.

    Computes a regular grid of point-source positions in the object
    plane and returns the positions along with the computed sensor
    distance.  **Render one point source at a time** to avoid PSF
    overlap — iterate over ``grid_positions`` and call
    :func:`setup_psf_scene` or manually place the source for each.

    Parameters
    ----------
    lens_obj : bpy.types.Object
        Fresnel lens object.
    grid_rows, grid_cols : int
        Number of grid positions along Y and X (default 5×5).
    grid_spacing : float
        Spacing between adjacent grid points in mm (default 20).
    object_distance : float
        Distance from lens to the object plane in mm.
    render_samples : int
        Render samples per point (passed to :func:`setup_psf_scene`).
    render_resolution : tuple[int, int]
        Render resolution per point.

    Returns
    -------
    dict
        Keys:

        - ``"grid_positions"`` — list of ``(x, y, z)`` tuples in
          world space (z = −object_distance).
        - ``"object_distance"`` — float, the object-plane distance.
        - ``"sensor_distance"`` — float, computed image-plane dist.
        - ``"grid_shape"`` — ``(grid_rows, grid_cols)``.
        - ``"render_samples"`` — int.
        - ``"render_resolution"`` — tuple.
        - ``"instructions"`` — human-readable usage note.
    """
    _require_bpy()
    from .scene_setup import _thin_lens_image_distance

    focal_length = float(
        lens_obj.get(
            "fresnel_focal_length",
            lens_obj.get("lens_focal_length", 200.0),
        )
    )
    sensor_distance = _thin_lens_image_distance(
        focal_length, object_distance
    )

    # Build grid centred on the optical axis
    positions: list[tuple[float, float, float]] = []
    for row in range(grid_rows):
        y = (row - (grid_rows - 1) / 2.0) * grid_spacing
        for col in range(grid_cols):
            x = (col - (grid_cols - 1) / 2.0) * grid_spacing
            positions.append((x, y, -object_distance))

    logger.info(
        "PSF grid: %d×%d (%d positions), spacing=%.1f mm, "
        "obj_dist=%.1f, sensor_dist=%.2f",
        grid_rows,
        grid_cols,
        len(positions),
        grid_spacing,
        object_distance,
        sensor_distance,
    )

    return {
        "grid_positions": positions,
        "object_distance": object_distance,
        "sensor_distance": sensor_distance,
        "grid_shape": (grid_rows, grid_cols),
        "render_samples": render_samples,
        "render_resolution": render_resolution,
        "instructions": (
            "Render ONE point source at a time to avoid PSF "
            "overlap.  For each position in 'grid_positions', "
            "call setup_psf_scene() with that position as "
            "'point_position', render, then extract the PSF "
            "with extract_psf_from_render()."
        ),
    }
