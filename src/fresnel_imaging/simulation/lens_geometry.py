"""Plano-convex and multi-element lens geometry builder.

Builds lens meshes from first principles using the Lensmaker's equation
and optical prescription tables, then revolves 2-D profiles into Blender
mesh objects via the shared ``_revolve_profile`` helper.

Lensmaker's equation (plano-convex, one flat surface)
-----------------------------------------------------
For a plano-convex lens with one flat (infinite radius) surface and one
spherical surface of radius *R*, refractive index *n*, and focal length *f*:

    1/f = (n − 1) / R   →   R = f · (n − 1)

The spherical sag at radial distance *r* from the optical axis is:

    sag(r) = R − sqrt(R² − r²)

Prescription format
-------------------
Multi-element lenses are specified as a list of surface dictionaries in
Zemax-style convention.  Each surface dict has keys:

- ``radius``    – radius of curvature (float; use ``math.inf`` for flat)
- ``thickness`` – axial distance to the next surface in mm
- ``ior``       – refractive index of the medium *following* this surface
                  (1.0 for air gaps)
- ``diameter``  – clear aperture diameter in mm

Consecutive surfaces with ``ior > 1`` belong to the same glass element.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

try:
    import bmesh
    import bpy

    _HAS_BPY = True
except ImportError:
    bpy = None  # type: ignore[assignment]
    bmesh = None  # type: ignore[assignment]
    _HAS_BPY = False

from .fresnel_lens import _revolve_profile
from .scene_setup import _thin_lens_image_distance  # noqa: F401

logger = logging.getLogger(__name__)

# ── Glass catalog: (n_d, Cauchy-B) at 587.6 nm ─────────────────────
GLASS_CATALOG: dict[str, tuple[float, float]] = {
    "BK7": (1.5168, 0.00420),
    "SF5": (1.6727, 0.00896),
    "LAK9": (1.6910, 0.00531),
    "FUSED_SILICA": (1.4585, 0.00354),
    "PMMA": (1.4917, 0.00000),
}


def _require_bpy() -> None:
    if not _HAS_BPY:
        raise RuntimeError(
            "This module must be run inside Blender. "
            "Use: blender -b -P <script.py>"
        )


# ── Plano-convex profile ───────────────────────────────────────────


def compute_planoconvex_profile(
    diameter: float,
    focal_length: float,
    thickness: float,
    material_ior: float = 1.5168,
    n_points: int = 64,
) -> list[tuple[float, float]]:
    """Compute the 2-D cross-section of a plano-convex lens.

    The flat (plano) face sits at z = 0 and the spherical cap faces
    toward +Z.  The profile is a closed loop suitable for revolution:
    centre-top → edge-top → edge-bottom → centre-bottom.

    Parameters
    ----------
    diameter : float
        Lens diameter in mm.
    focal_length : float
        Focal length in mm.
    thickness : float
        Centre thickness (axial) in mm.
    material_ior : float
        Refractive index (default BK7 = 1.5168).
    n_points : int
        Number of radial samples on the curved face (default 64).

    Returns
    -------
    list[tuple[float, float]]
        ``(r, z)`` coordinates forming a closed revolution profile.

    Raises
    ------
    ValueError
        If the semi-diameter exceeds the radius of curvature.
    """
    r_max = diameter / 2.0
    radius_of_curvature = focal_length * (material_ior - 1.0)

    if r_max >= radius_of_curvature:
        raise ValueError(
            f"Semi-diameter ({r_max:.3f} mm) must be less than "
            f"the radius of curvature ({radius_of_curvature:.3f}"
            f" mm). Increase focal_length or decrease diameter."
        )

    def _sag(r: float) -> float:
        return radius_of_curvature - math.sqrt(
            radius_of_curvature**2 - r**2
        )

    sag_edge = _sag(r_max)

    profile: list[tuple[float, float]] = []

    # ── Curved (top) face: centre → edge ────────────────────────
    for i in range(n_points):
        r = r_max * i / (n_points - 1)
        z = thickness + _sag(r) - sag_edge
        profile.append((r, z))

    # ── Flat (bottom) face: edge → centre ───────────────────────
    profile.append((r_max, 0.0))
    profile.append((0.0, 0.0))

    return profile


# ── Blender plano-convex lens ──────────────────────────────────────


def create_planoconvex_lens(
    diameter: float = 50.0,
    focal_length: float = 100.0,
    thickness: float = 5.0,
    material_ior: float = 1.5168,
    name: str = "PlanoConvexLens",
    segments: int = 128,
) -> Any:
    """Create a plano-convex lens mesh in the current Blender scene.

    Parameters
    ----------
    diameter, focal_length, thickness, material_ior
        Optical parameters (mm / dimensionless).
    name : str
        Blender object name.
    segments : int
        Angular segments for revolution (default 128).

    Returns
    -------
    bpy.types.Object
    """
    _require_bpy()

    profile = compute_planoconvex_profile(
        diameter=diameter,
        focal_length=focal_length,
        thickness=thickness,
        material_ior=material_ior,
    )

    obj = _revolve_profile(profile, segments=segments, name=name)

    obj["lens_diameter"] = diameter
    obj["lens_focal_length"] = focal_length
    obj["lens_thickness"] = thickness
    obj["lens_material_ior"] = material_ior

    logger.info(
        "Created plano-convex lens '%s' "
        "(D=%.1f, f=%.1f, t=%.1f, n=%.4f)",
        name,
        diameter,
        focal_length,
        thickness,
        material_ior,
    )

    return obj


# ── Prescription validation ────────────────────────────────────────

_REQUIRED_KEYS = {"radius", "thickness", "ior", "diameter"}


def validate_prescription(
    prescription: list[dict[str, float]],
) -> None:
    """Validate an optical prescription table.

    Raises
    ------
    ValueError
        If any surface is missing required keys, has non-positive
        thickness (except the last surface), or has an invalid IOR
        for a glass medium.
    """
    if not prescription:
        raise ValueError("Prescription must contain at least one surface.")

    for idx, surface in enumerate(prescription):
        missing = _REQUIRED_KEYS - surface.keys()
        if missing:
            raise ValueError(
                f"Surface {idx}: missing required keys "
                f"{sorted(missing)}."
            )

        is_last = idx == len(prescription) - 1
        if not is_last and surface["thickness"] <= 0:
            raise ValueError(
                f"Surface {idx}: thickness must be positive "
                f"(got {surface['thickness']})."
            )
        if is_last and surface["thickness"] < 0:
            raise ValueError(
                f"Surface {idx}: thickness must be >= 0 "
                f"(got {surface['thickness']})."
            )

        if surface["ior"] > 1.0:
            # Glass medium – IOR looks fine
            pass
        elif surface["ior"] < 1.0:
            raise ValueError(
                f"Surface {idx}: IOR must be >= 1.0 "
                f"(got {surface['ior']})."
            )


# ── Multi-element lens from prescription ───────────────────────────


def _surface_sag(
    r: float,
    radius_of_curvature: float,
) -> float:
    """Spherical sag at radial distance *r*."""
    if math.isinf(radius_of_curvature):
        return 0.0
    sign = 1.0 if radius_of_curvature > 0 else -1.0
    roc_abs = abs(radius_of_curvature)
    return sign * (roc_abs - math.sqrt(roc_abs**2 - r**2))


def _element_profile(
    front_radius: float,
    back_radius: float,
    thickness: float,
    diameter: float,
    n_points: int = 64,
) -> list[tuple[float, float]]:
    """Build a closed (r, z) profile for a single lens element.

    Front surface at z = thickness, back surface at z = 0.
    """
    r_max = diameter / 2.0

    front_sag_edge = _surface_sag(r_max, front_radius)
    back_sag_edge = _surface_sag(r_max, back_radius)

    profile: list[tuple[float, float]] = []

    # Front face (centre → edge)
    for i in range(n_points):
        r = r_max * i / (n_points - 1)
        sag = _surface_sag(r, front_radius)
        z = thickness + sag - front_sag_edge
        profile.append((r, z))

    # Back face (edge → centre)
    for i in range(n_points - 1, -1, -1):
        r = r_max * i / (n_points - 1)
        sag = _surface_sag(r, back_radius)
        z = sag - back_sag_edge
        profile.append((r, z))

    return profile


def create_lens_from_prescription(
    prescription: list[dict[str, float]],
    name: str = "CustomLens",
    segments: int = 128,
) -> list[Any]:
    """Create multi-element lens meshes from a prescription table.

    Each pair of consecutive surfaces sharing a glass medium
    (``ior > 1``) is treated as one lens element.  Elements are
    positioned along the Z axis using cumulative thickness offsets.

    Parameters
    ----------
    prescription : list[dict]
        Surface list in Zemax convention (see module docstring).
    name : str
        Base name for Blender objects.
    segments : int
        Angular segments for revolution.

    Returns
    -------
    list[Any]
        One Blender object per glass element.
    """
    _require_bpy()
    from .materials import assign_material, create_glass_material

    validate_prescription(prescription)

    elements: list[Any] = []
    z_offset = 0.0
    idx = 0

    while idx < len(prescription) - 1:
        surf = prescription[idx]
        next_surf = prescription[idx + 1]

        if surf["ior"] > 1.0:
            # Glass element: front = surf, back = next_surf
            front_r = surf["radius"]
            back_r = next_surf["radius"]
            thick = surf["thickness"]
            diam = min(surf["diameter"], next_surf["diameter"])

            elem_name = f"{name}_E{len(elements)}"
            profile = _element_profile(
                front_radius=front_r,
                back_radius=back_r,
                thickness=thick,
                diameter=diam,
            )

            obj = _revolve_profile(
                profile, segments=segments, name=elem_name,
            )
            obj.location = (0.0, 0.0, z_offset)

            # Assign glass material
            ior = surf["ior"]
            cauchy_b = 0.0
            for gname, (n_d, cb) in GLASS_CATALOG.items():
                if abs(n_d - ior) < 0.002:
                    cauchy_b = cb
                    break
            mat = create_glass_material(
                name=f"{elem_name}_Glass",
                ior=ior,
                cauchy_b=cauchy_b,
            )
            assign_material(obj, mat)

            obj["lens_diameter"] = diam
            obj["lens_focal_length"] = 0.0
            obj["lens_thickness"] = thick
            obj["lens_material_ior"] = ior

            elements.append(obj)

            logger.info(
                "Element %d: R1=%.2f R2=%.2f t=%.2f n=%.4f",
                len(elements) - 1,
                front_r,
                back_r,
                thick,
                ior,
            )

            z_offset += thick
            idx += 1  # skip next_surf (back face consumed)
        else:
            # Air gap
            z_offset += surf["thickness"]

        idx += 1

    # Store full prescription as JSON on each element
    rx_json = json.dumps(prescription)
    for obj in elements:
        obj["lens_prescription"] = rx_json

    logger.info(
        "Created %d element(s) for '%s'", len(elements), name,
    )

    return elements
