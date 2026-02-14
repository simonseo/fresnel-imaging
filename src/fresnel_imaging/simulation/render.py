"""Rendering automation for the Fresnel lens simulation pipeline.

Supports LuxCoreRender (preferred for caustics through glass) with
automatic fallback to Cycles.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
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


def configure_luxcore_render(
    samples: int = 2000,
    engine: str = "BIDIR",
    resolution: tuple[int, int] = (1920, 1080),
    light_depth: int = 15,
    path_depth: int = 15,
    sampler: str = "METROPOLIS",
    use_photongi: bool = False,
    photongi_count: float = 20.0,
) -> None:
    """Configure LuxCoreRender settings for glass-caustic rendering.

    Parameters
    ----------
    samples : int
        Halt sample count (default 2000 for low noise through glass).
    engine : str
        ``"BIDIR"`` (bidirectional, best for caustics) or ``"PATH"``.
    resolution : tuple[int, int]
        ``(width, height)`` in pixels.
    light_depth, path_depth : int
        Max bounce depth for light / camera rays.
    sampler : str
        ``"METROPOLIS"`` (default, good for caustics), ``"SOBOL"``, or
        ``"RANDOM"``.
    use_photongi : bool
        Enable PhotonGI cache (PATH engine only).
    photongi_count : float
        Photon count in millions (default 20).
    """
    _require_bpy()

    scene = bpy.context.scene

    if not _is_luxcore_available():
        print("LuxCore not available — configuring Cycles instead.")
        _configure_cycles_fallback(samples, resolution)
        return

    scene.render.engine = "LUXCORE"

    config = scene.luxcore.config
    config.engine = engine
    config.sampler = sampler

    if engine == "BIDIR":
        config.bidir_light_maxdepth = light_depth
        config.bidir_path_maxdepth = path_depth
    else:
        config.path_maxdepth = max(light_depth, path_depth)
        if use_photongi:
            config.photongi.enabled = True
            config.photongi.photon_maxcount = photongi_count
            config.photongi.photon_maxdepth = path_depth

    halt = scene.luxcore.halt
    halt.enable = True
    halt.use_samples = True
    halt.samples = samples

    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100


def _configure_cycles_fallback(
    samples: int,
    resolution: tuple[int, int],
) -> None:
    """Configure Cycles as a fallback renderer."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    scene.cycles.samples = samples
    scene.cycles.max_bounces = 32
    scene.cycles.glossy_bounces = 16
    scene.cycles.transmission_bounces = 16
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True

    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100


def render_scene(
    output_path: str,
    blend_file: str | None = None,
) -> str:
    """Render the current scene to *output_path*.

    If *blend_file* is given and we are not already inside that file,
    open it first.

    Parameters
    ----------
    output_path : str
        Destination file path (e.g. ``"output/render.png"``).
    blend_file : str | None
        Optional ``.blend`` file to open before rendering.

    Returns
    -------
    str
        The absolute path to the rendered image.
    """
    _require_bpy()

    if blend_file is not None:
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend_file))

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    scene = bpy.context.scene
    scene.render.filepath = abs_output
    scene.render.image_settings.file_format = _format_from_ext(abs_output)

    bpy.ops.render.render(write_still=True)

    return abs_output


def _format_from_ext(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".exr": "OPEN_EXR",
        ".tiff": "TIFF",
        ".tif": "TIFF",
        ".bmp": "BMP",
    }.get(ext, "PNG")


@dataclass
class RenderConfig:
    """A single render configuration for batch rendering."""

    output_name: str
    samples: int = 2000
    engine: str = "BIDIR"
    resolution: tuple[int, int] = (1920, 1080)
    extra: dict[str, Any] = field(default_factory=dict)


def batch_render(
    blend_file: str,
    output_dir: str,
    configs: list[RenderConfig],
) -> list[str]:
    """Render multiple configurations from a single blend file.

    Each config produces a separate output image.

    Parameters
    ----------
    blend_file : str
        Path to the ``.blend`` file.
    output_dir : str
        Directory for output images.
    configs : list[RenderConfig]
        List of render configurations.

    Returns
    -------
    list[str]
        Paths to rendered images.
    """
    _require_bpy()

    abs_blend = os.path.abspath(blend_file)
    abs_out = os.path.abspath(output_dir)
    os.makedirs(abs_out, exist_ok=True)

    results: list[str] = []

    for cfg in configs:
        bpy.ops.wm.open_mainfile(filepath=abs_blend)

        configure_luxcore_render(
            samples=cfg.samples,
            engine=cfg.engine,
            resolution=cfg.resolution,
        )

        out_path = os.path.join(abs_out, cfg.output_name)
        rendered = render_scene(out_path)
        results.append(rendered)

    return results


def run_headless(
    script_path: str,
    blend_file: str | None = None,
    blender_bin: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build the command line for running a Blender script headlessly.

    Does **not** execute the command — returns the argv list so the
    caller can run it with ``subprocess`` or inspect it.

    Parameters
    ----------
    script_path : str
        Path to the Python script to execute inside Blender.
    blend_file : str | None
        Optional ``.blend`` file to open.
    blender_bin : str | None
        Path to the Blender binary. If *None*, searches ``$PATH`` and
        common installation directories.
    extra_args : list[str] | None
        Additional arguments appended after ``--``.

    Returns
    -------
    list[str]
        Command-line argument list.
    """
    blender = blender_bin or _find_blender()

    cmd: list[str] = [blender, "-b"]

    if blend_file is not None:
        cmd.append(os.path.abspath(blend_file))

    cmd.extend(["-P", os.path.abspath(script_path)])

    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    return cmd


def _find_blender() -> str:
    """Locate the Blender binary on the system."""
    found = shutil.which("blender")
    if found:
        return found

    # Common installation paths
    candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
        "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
        "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find Blender binary. Set the 'blender_bin' argument "
        "or add Blender to your PATH."
    )


def execute_headless(
    script_path: str,
    blend_file: str | None = None,
    blender_bin: str | None = None,
    extra_args: list[str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Blender script headlessly and return the result.

    Parameters
    ----------
    script_path : str
        Path to the Python script.
    blend_file : str | None
        Optional blend file.
    blender_bin : str | None
        Blender binary path.
    extra_args : list[str] | None
        Extra arguments after ``--``.
    timeout : int | None
        Timeout in seconds.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd = run_headless(script_path, blend_file, blender_bin, extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
