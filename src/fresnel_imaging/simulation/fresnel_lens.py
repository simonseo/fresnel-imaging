"""Procedural Fresnel lens geometry generator using Blender's bmesh API.

Generates an optically-accurate plano-convex Fresnel lens by computing the
groove prism angles from Snell's law and revolving the resulting sawtooth
cross-section profile around the optical axis.

Physics
-------
For a plano-convex Fresnel lens (flat back, grooved front) with refractive
index *n* and focal length *f*, a groove at radial distance *r* must deflect
a paraxial ray parallel to the optical axis so that it passes through the
focal point.  The required deflection angle is ``delta = arctan(r / f)``
and the groove face angle *alpha* (measured from the flat plane) satisfies
Snell's law at the air-glass interface:

    n * sin(alpha) = sin(alpha + delta)

Solving for *alpha*:

    alpha = arctan(sin(delta) / (n - cos(delta)))
"""

from __future__ import annotations

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


def _require_bpy() -> None:
    if not _HAS_BPY:
        raise RuntimeError(
            "This module must be run inside Blender. "
            "Use: blender -b -P <script.py>"
        )


def _groove_face_angle(r: float, f: float, n: float) -> float:
    """Return the groove face angle (radians) for a groove at radius *r*.

    Solves  n·sin(α) = sin(α + δ)  where  δ = arctan(r / f).
    """
    # Deflection angle: ray at radius r must bend toward focal point
    delta = math.atan2(r, f)
    # Snell's law rearranged:
    #   n·sin(α) = sin(α)·cos(δ) + cos(α)·sin(δ)
    #   sin(α)·(n - cos(δ)) = cos(α)·sin(δ)
    #   tan(α) = sin(δ) / (n - cos(δ))
    alpha = math.atan2(math.sin(delta), n - math.cos(delta))
    return alpha


def compute_profile(
    diameter: float,
    focal_length: float,
    n_grooves: int,
    thickness: float,
    material_ior: float = 1.5168,
) -> list[tuple[float, float]]:
    """Compute the 2-D cross-section profile of the Fresnel lens.

    Returns a list of ``(r, z)`` coordinates describing the sawtooth profile
    from the centre of the lens (r=0) to the outer edge (r=diameter/2).
    The flat (back) face sits at z=0; grooves protrude into positive z.

    Parameters
    ----------
    diameter : float
        Lens diameter in mm.
    focal_length : float
        Focal length in mm.
    n_grooves : int
        Number of concentric annular grooves.
    thickness : float
        Total lens thickness in mm (base plate + tallest groove).
    material_ior : float
        Refractive index of the lens material (default BK7 = 1.5168).
    """
    radius = diameter / 2.0
    groove_width = radius / n_grooves
    # Reserve a fraction of thickness for the base plate
    base_z = thickness * 0.3
    max_groove_height = thickness - base_z

    profile: list[tuple[float, float]] = []
    # Centre point on the flat base
    profile.append((0.0, base_z))

    for i in range(n_grooves):
        r_inner = i * groove_width
        r_outer = (i + 1) * groove_width
        r_mid = (r_inner + r_outer) / 2.0

        alpha = _groove_face_angle(r_mid, focal_length, material_ior)
        groove_height = min(groove_width * math.tan(alpha), max_groove_height)

        if i == 0:
            # First groove starts from the centre axis
            profile.append((r_inner, base_z))
        # Groove rises along the refracting face
        profile.append((r_outer, base_z + groove_height))
        # Vertical drop back to base (draft face)
        profile.append((r_outer, base_z))

    # Close the profile along the outer edge and back via the flat face
    profile.append((radius, 0.0))
    profile.append((0.0, 0.0))

    return profile


def _revolve_profile(
    profile: list[tuple[float, float]],
    segments: int,
    name: str,
) -> Any:
    """Revolve a 2-D (r, z) profile around the z-axis and return a Blender object."""
    _require_bpy()

    bm = bmesh.new()

    # Build profile vertices (in the XZ plane, y=0)
    profile_verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]

    # Connect consecutive verts with edges
    profile_edges = [
        bm.edges.new((profile_verts[i], profile_verts[i + 1]))
        for i in range(len(profile_verts) - 1)
    ]

    geom = list(profile_verts) + list(profile_edges)
    bmesh.ops.spin(
        bm,
        geom=geom,
        angle=math.pi * 2,
        steps=segments,
        axis=(0.0, 0.0, 1.0),
        cent=(0.0, 0.0, 0.0),
        use_merge=True,
        use_normal_flip=False,
        use_duplicate=False,
    )

    # Remove duplicate vertices introduced by the merge seam
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()

    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj


def create_fresnel_lens(
    diameter: float = 100.0,
    focal_length: float = 200.0,
    n_grooves: int = 50,
    thickness: float = 2.0,
    material_ior: float = 1.5168,
    name: str = "FresnelLens",
    segments: int = 128,
) -> Any:
    """Create a procedural Fresnel lens mesh in the current Blender scene.

    The lens is centred at the origin with the optical axis along +Z.
    The flat (plano) face sits at z=0 and grooves face toward +Z.

    Parameters
    ----------
    diameter : float
        Lens diameter in mm (default 100).
    focal_length : float
        Focal length in mm (default 200).
    n_grooves : int
        Number of concentric grooves (default 50).
    thickness : float
        Total lens thickness in mm (default 2.0).
    material_ior : float
        Refractive index of the lens material (default BK7 = 1.5168).
    name : str
        Blender object name (default ``"FresnelLens"``).
    segments : int
        Number of angular segments for the revolution (default 128).

    Returns
    -------
    bpy.types.Object
        The created Blender mesh object.
    """
    _require_bpy()

    profile = compute_profile(
        diameter=diameter,
        focal_length=focal_length,
        n_grooves=n_grooves,
        thickness=thickness,
        material_ior=material_ior,
    )

    obj = _revolve_profile(profile, segments=segments, name=name)

    # Store lens parameters as custom properties for downstream use
    obj["fresnel_diameter"] = diameter
    obj["fresnel_focal_length"] = focal_length
    obj["fresnel_n_grooves"] = n_grooves
    obj["fresnel_thickness"] = thickness
    obj["fresnel_material_ior"] = material_ior

    return obj


if __name__ == "__main__":
    _require_bpy()

    # Clear default scene objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    lens = create_fresnel_lens(
        diameter=100.0,
        focal_length=200.0,
        n_grooves=50,
        thickness=2.0,
    )
    print(f"Created Fresnel lens: {lens.name}")
    print(f"  Vertices: {len(lens.data.vertices)}")
    print(f"  Faces:    {len(lens.data.polygons)}")
