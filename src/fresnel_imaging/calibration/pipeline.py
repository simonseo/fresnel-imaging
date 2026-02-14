"""End-to-end PSF calibration pipeline for Fresnel lens imaging.

Orchestrates distortion correction, alignment, and patchwise PSF estimation
into a single calibration workflow.  Results can be saved / loaded and applied
to novel images via spatially-varying deconvolution.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .alignment import extract_patches, stitch_patches
from .distortion import calibrate_from_multiple, estimate_distortion, undistort_image
from .psf_estimation import (
    estimate_spatially_varying_psf,
    interpolate_psf,
)

LOGGER = logging.getLogger(__name__)


def calibrate_fresnel_lens(
    blurred_image: np.ndarray,
    reference_image: np.ndarray,
    pattern_size: tuple[int, int] = (8, 6),
    psf_size: int = 31,
    patch_size: int = 64,
    method: str = "qp",
    per_channel: bool = True,
    undistort: bool = True,
    manual_corners: NDArray[np.floating] | None = None,
) -> dict[str, object]:
    """Run the full Fresnel lens calibration pipeline.

    Steps:
    1.  Distortion correction — detect checkerboard corners in the blurred
        image, compute a camera matrix and distortion coefficients, then
        undistort both blurred and reference images.
    2.  Patchwise PSF estimation — align images globally then locally per
        patch, estimate a PSF kernel in each patch using the selected method.

    Parameters
    ----------
    blurred_image : np.ndarray
        Image captured through the Fresnel lens.
    reference_image : np.ndarray
        Corresponding sharp reference (e.g. rendered ground truth).
    pattern_size : tuple[int, int], optional
        Inner checkerboard corner count ``(columns, rows)``.
    psf_size : int, optional
        Square PSF support size (must be odd).
    patch_size : int, optional
        Patch size used for local PSF estimation.
    method : str, optional
        PSF estimation method: ``"qp"``, ``"wiener"``, or ``"rl"``.
    per_channel : bool, optional
        Estimate per-channel PSFs for colour images.
    undistort : bool, optional
        If True, attempt distortion correction before PSF estimation.
    manual_corners : NDArray[np.floating] | None, optional
        Pre-detected checkerboard corners.  Pass this when automatic corner
        detection fails due to severe Fresnel distortion.

    Returns
    -------
    dict[str, object]
        Calibration result dictionary with keys:

        - ``"camera_matrix"`` — ``(3, 3)`` camera intrinsic matrix or *None*.
        - ``"dist_coeffs"`` — distortion coefficients or *None*.
        - ``"psf_grid"`` — list (or list-of-lists for per-channel) of PSFs.
        - ``"positions"`` — patch-centre coordinates ``(row, col)``.
        - ``"undistorted"`` — whether distortion correction was applied.
    """
    camera_matrix: NDArray[np.floating] | None = None
    dist_coeffs: NDArray[np.floating] | None = None
    did_undistort = False

    blurred = blurred_image.copy()
    reference = reference_image.copy()

    # --- Step 1: Distortion correction ---
    if undistort:
        if manual_corners is not None:
            result = calibrate_from_multiple(
                [blurred],
                pattern_size=pattern_size,
                manual_corners=[manual_corners],
            )
            if result is not None:
                camera_matrix = result.camera_matrix
                dist_coeffs = result.dist_coeffs
        else:
            camera_matrix, dist_coeffs = estimate_distortion(blurred, pattern_size)

        if camera_matrix is not None and dist_coeffs is not None:
            blurred = undistort_image(blurred, camera_matrix, dist_coeffs)
            reference = undistort_image(reference, camera_matrix, dist_coeffs)
            did_undistort = True
            LOGGER.info("Distortion correction applied successfully.")
        else:
            LOGGER.warning(
                "Distortion correction skipped — checkerboard corners not found. "
                "Consider providing manual_corners."
            )

    # --- Step 2: Patchwise PSF estimation ---
    psf_grid, positions = estimate_spatially_varying_psf(
        blurred,
        reference,
        patch_size=patch_size,
        psf_size=psf_size,
        overlap=0.5,
        method=method,
        per_channel=per_channel,
    )

    LOGGER.info(
        "PSF calibration complete: %d patches, method=%s, per_channel=%s.",
        len(positions),
        method,
        per_channel,
    )

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "psf_grid": psf_grid,
        "positions": positions,
        "undistorted": did_undistort,
    }


def apply_spatially_varying_deconv(
    image: np.ndarray,
    psf_grid: list[np.ndarray] | list[list[np.ndarray]],
    positions: list[tuple[int, int]],
    deconv_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    patch_size: int = 64,
    overlap: float = 0.5,
) -> np.ndarray:
    """Apply spatially-varying deconvolution using an estimated PSF grid.

    For each image patch the locally interpolated PSF is fed to ``deconv_fn``.
    The deconvolved patches are then stitched with feathered blending.

    Parameters
    ----------
    image : np.ndarray
        Blurred input image (grayscale or colour).
    psf_grid : list[np.ndarray] | list[list[np.ndarray]]
        PSF grid from :func:`calibrate_fresnel_lens`.  For per-channel
        calibration this is a list of lists (one inner list per channel).
    positions : list[tuple[int, int]]
        Patch-centre positions corresponding to ``psf_grid``.
    deconv_fn : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Deconvolution function accepting ``(blurred_patch, psf)`` and
        returning the restored patch.
    patch_size : int, optional
        Patch size consistent with the calibration step.
    overlap : float, optional
        Patch overlap fraction.

    Returns
    -------
    np.ndarray
        Deconvolved image.
    """
    is_per_channel = (
        isinstance(psf_grid, list)
        and len(psf_grid) > 0
        and isinstance(psf_grid[0], list)
    )

    if is_per_channel and image.ndim == 3 and image.shape[2] == 3:
        channels: list[np.ndarray] = []
        for c_idx in range(3):
            channel_psf_grid: list[np.ndarray] = psf_grid[c_idx]  # type: ignore[index]
            channel_img = image[:, :, c_idx]
            deconvolved = _deconv_single_channel(
                channel_img,
                channel_psf_grid,
                positions,
                deconv_fn,
                patch_size,
                overlap,
            )
            channels.append(deconvolved)
        return np.stack(channels, axis=-1)

    flat_grid: list[np.ndarray]
    if is_per_channel:
        flat_grid = psf_grid[0]  # type: ignore[assignment]
    else:
        flat_grid = psf_grid  # type: ignore[assignment]

    if image.ndim == 3:
        channels = []
        for c_idx in range(image.shape[2]):
            deconvolved = _deconv_single_channel(
                image[:, :, c_idx],
                flat_grid,
                positions,
                deconv_fn,
                patch_size,
                overlap,
            )
            channels.append(deconvolved)
        return np.stack(channels, axis=-1)

    return _deconv_single_channel(
        image, flat_grid, positions, deconv_fn, patch_size, overlap
    )


def save_calibration(
    filepath: str | Path,
    camera_matrix: NDArray[np.floating] | None,
    dist_coeffs: NDArray[np.floating] | None,
    psf_grid: list[np.ndarray] | list[list[np.ndarray]],
    positions: list[tuple[int, int]],
) -> None:
    """Save calibration data to an ``.npz`` file.

    Parameters
    ----------
    filepath : str | Path
        Destination path (will be given an ``.npz`` suffix automatically).
    camera_matrix : NDArray[np.floating] | None
        Camera intrinsic matrix, or *None*.
    dist_coeffs : NDArray[np.floating] | None
        Distortion coefficients, or *None*.
    psf_grid : list[np.ndarray] | list[list[np.ndarray]]
        PSF grid (flat list or per-channel nested list).
    positions : list[tuple[int, int]]
        Patch-centre positions.
    """
    save_dict: dict[str, np.ndarray] = {}

    if camera_matrix is not None:
        save_dict["camera_matrix"] = np.asarray(camera_matrix)
    if dist_coeffs is not None:
        save_dict["dist_coeffs"] = np.asarray(dist_coeffs)

    save_dict["positions"] = np.array(positions, dtype=np.int64)

    is_per_channel = (
        isinstance(psf_grid, list)
        and len(psf_grid) > 0
        and isinstance(psf_grid[0], list)
    )

    if is_per_channel:
        save_dict["per_channel"] = np.array([1])
        for c_idx, channel_psfs in enumerate(psf_grid):
            for p_idx, psf in enumerate(channel_psfs):  # type: ignore[union-attr]
                save_dict[f"psf_c{c_idx}_p{p_idx}"] = np.asarray(psf)
        save_dict["n_channels"] = np.array([len(psf_grid)])
        save_dict["n_patches"] = np.array(
            [len(channel_psfs) for channel_psfs in psf_grid]  # type: ignore[union-attr]
        )
    else:
        save_dict["per_channel"] = np.array([0])
        for p_idx, psf in enumerate(psf_grid):
            save_dict[f"psf_p{p_idx}"] = np.asarray(psf)  # type: ignore[arg-type]
        save_dict["n_patches"] = np.array([len(psf_grid)])

    np.savez(filepath, **save_dict)
    LOGGER.info("Calibration saved to %s.", filepath)


def load_calibration(
    filepath: str | Path,
) -> dict[str, object]:
    """Load calibration data from an ``.npz`` file.

    Parameters
    ----------
    filepath : str | Path
        Path to the ``.npz`` calibration file.

    Returns
    -------
    dict[str, object]
        Dictionary with keys ``"camera_matrix"``, ``"dist_coeffs"``,
        ``"psf_grid"``, and ``"positions"``.
    """
    data = np.load(filepath, allow_pickle=False)

    camera_matrix = data["camera_matrix"] if "camera_matrix" in data else None
    dist_coeffs = data["dist_coeffs"] if "dist_coeffs" in data else None
    positions = [tuple(int(v) for v in row) for row in data["positions"]]

    is_per_channel = bool(data["per_channel"][0])

    if is_per_channel:
        n_channels = int(data["n_channels"][0])
        n_patches_per_ch = data["n_patches"]
        psf_grid: list[np.ndarray] | list[list[np.ndarray]] = []
        for c_idx in range(n_channels):
            channel_psfs: list[np.ndarray] = []
            n_patches = int(n_patches_per_ch[c_idx])
            for p_idx in range(n_patches):
                channel_psfs.append(data[f"psf_c{c_idx}_p{p_idx}"])
            psf_grid.append(channel_psfs)  # type: ignore[arg-type]
    else:
        n_patches = int(data["n_patches"][0])
        psf_grid = [data[f"psf_p{p_idx}"] for p_idx in range(n_patches)]

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "psf_grid": psf_grid,
        "positions": positions,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deconv_single_channel(
    image: np.ndarray,
    psf_grid: list[np.ndarray],
    positions: list[tuple[int, int]],
    deconv_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    patch_size: int,
    overlap: float,
) -> np.ndarray:
    """Deconvolve a single-channel image using interpolated local PSFs."""
    patches, origins = extract_patches(image, patch_size, overlap)
    deconvolved_patches: list[np.ndarray] = []

    for patch, origin in zip(patches, origins):
        centre = (
            origin[0] + patch.shape[0] / 2.0,
            origin[1] + patch.shape[1] / 2.0,
        )
        local_psf = interpolate_psf(psf_grid, positions, centre)

        try:
            restored = deconv_fn(patch.astype(np.float64), local_psf)
        except Exception:
            LOGGER.warning(
                "Deconvolution failed at patch origin %s; using input patch.", origin,
            )
            restored = patch

        deconvolved_patches.append(np.asarray(restored))

    return stitch_patches(deconvolved_patches, origins, image.shape, overlap)
