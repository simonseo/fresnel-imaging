"""Blender-based Fresnel lens optical simulation pipeline.

Modules
-------
fresnel_lens
    Procedural Fresnel lens geometry generation via bmesh.
lens_geometry
    Plano-convex and multi-element lens geometry builders.
materials
    LuxCoreRender / Cycles material setup (glass, diffuse).
psf_measurement
    PSF measurement automation via point source rendering.
scene_setup
    Calibration and imaging scene configuration.
render
    Rendering automation and headless execution.

All modules guard ``import bpy`` behind try/except so they can be imported
outside Blender for documentation, type-checking, and testing purposes.
"""

# bpy is only available inside Blender's embedded Python.
# Lazy imports prevent ImportError when this package is imported externally.

from importlib import import_module


def __getattr__(name: str):  # noqa: N807
    """Lazy-load simulation submodules to avoid bpy import failures."""
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "fresnel_lens",
    "lens_geometry",
    "materials",
    "psf_measurement",
    "render",
    "scene_setup",
]
