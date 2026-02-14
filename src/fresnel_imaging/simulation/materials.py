"""LuxCoreRender and Cycles material setup for Fresnel lens simulation.

Provides glass materials with correct IOR and Cauchy-B dispersion for
BK7 and other optical glasses.  Falls back to Cycles Glass BSDF when
BlendLuxCore is not installed.
"""

from __future__ import annotations

from typing import Any

try:
    import bpy

    _HAS_BPY = True
except ImportError:
    bpy = None  # type: ignore[assignment]
    _HAS_BPY = False


def _require_bpy() -> None:
    if not _HAS_BPY:
        raise RuntimeError(
            "This module must be run inside Blender. "
            "Use: blender -b -P <script.py>"
        )


def _is_luxcore_available() -> bool:
    if not _HAS_BPY:
        return False
    return bpy.context.scene.render.engine == "LUXCORE" or hasattr(
        bpy.context.scene, "luxcore"
    )


def _create_luxcore_glass(
    name: str,
    ior: float,
    cauchy_b: float,
    transmission_color: tuple[float, float, float],
    reflection_color: tuple[float, float, float],
) -> Any:
    """Create a LuxCoreRender glass material with dispersion."""
    mat = bpy.data.materials.new(name=name)

    node_tree = bpy.data.node_groups.new(
        name=f"{name}_Nodes", type="luxcore_material_nodes"
    )
    mat.luxcore.node_tree = node_tree

    output = node_tree.nodes.new("LuxCoreNodeMatOutput")
    output.location = (300, 200)

    glass = node_tree.nodes.new("LuxCoreNodeMatGlass")
    glass.location = (0, 200)

    glass.inputs["IOR"].default_value = ior
    glass.inputs["Dispersion"].default_value = cauchy_b
    glass.inputs["Transmission Color"].default_value = transmission_color
    glass.inputs["Reflection Color"].default_value = reflection_color

    node_tree.links.new(glass.outputs["Material"], output.inputs["Material"])

    return mat


def _create_cycles_glass(
    name: str,
    ior: float,
    color: tuple[float, float, float],
) -> Any:
    """Fallback: create a Cycles Glass BSDF material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    glass_bsdf = tree.nodes.new("ShaderNodeBsdfGlass")
    glass_bsdf.location = (0, 0)
    glass_bsdf.inputs["IOR"].default_value = ior
    glass_bsdf.inputs["Color"].default_value = (*color, 1.0)
    glass_bsdf.inputs["Roughness"].default_value = 0.0

    tree.links.new(glass_bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


def create_glass_material(
    name: str = "BK7_Glass",
    ior: float = 1.5168,
    cauchy_b: float = 0.00420,
    transmission_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    reflection_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Any:
    """Create a glass material suitable for optical simulation.

    Attempts to create a LuxCoreRender glass material with Cauchy-B
    dispersion.  If LuxCore is unavailable, falls back to a Cycles
    Glass BSDF (without dispersion).

    Parameters
    ----------
    name : str
        Material name.
    ior : float
        Index of refraction (default BK7 = 1.5168).
    cauchy_b : float
        Cauchy B dispersion coefficient (default BK7 ≈ 0.00420).
        Only used with LuxCore; Cycles Glass BSDF has no dispersion.
    transmission_color, reflection_color : tuple[float, float, float]
        RGB colours for LuxCore glass (default white).

    Returns
    -------
    bpy.types.Material
    """
    _require_bpy()

    try:
        if _is_luxcore_available():
            mat = _create_luxcore_glass(
                name, ior, cauchy_b, transmission_color, reflection_color
            )
            mat["renderer"] = "luxcore"
            return mat
    except Exception:
        pass

    mat = _create_cycles_glass(name, ior, transmission_color)
    mat["renderer"] = "cycles"
    return mat


def create_diffuse_material(
    name: str,
    color: tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> Any:
    """Create a simple diffuse material.

    Uses a LuxCore matte node when available, otherwise a Cycles
    Diffuse BSDF.

    Parameters
    ----------
    name : str
        Material name.
    color : tuple[float, float, float]
        RGB diffuse colour.

    Returns
    -------
    bpy.types.Material
    """
    _require_bpy()

    try:
        if _is_luxcore_available():
            return _create_luxcore_matte(name, color)
    except Exception:
        pass

    return _create_cycles_diffuse(name, color)


def _create_luxcore_matte(
    name: str,
    color: tuple[float, float, float],
) -> Any:
    mat = bpy.data.materials.new(name=name)

    node_tree = bpy.data.node_groups.new(
        name=f"{name}_Nodes", type="luxcore_material_nodes"
    )
    mat.luxcore.node_tree = node_tree

    output = node_tree.nodes.new("LuxCoreNodeMatOutput")
    output.location = (300, 200)

    matte = node_tree.nodes.new("LuxCoreNodeMatMatte")
    matte.location = (0, 200)
    matte.inputs["Diffuse Color"].default_value = color

    node_tree.links.new(matte.outputs["Material"], output.inputs["Material"])

    return mat


def _create_cycles_diffuse(
    name: str,
    color: tuple[float, float, float],
) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.location = (0, 0)
    diffuse.inputs["Color"].default_value = (*color, 1.0)

    tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

    return mat


def assign_material(obj: Any, mat: Any) -> None:
    """Assign *mat* to *obj*, replacing any existing materials."""
    _require_bpy()
    obj.data.materials.clear()
    obj.data.materials.append(mat)
