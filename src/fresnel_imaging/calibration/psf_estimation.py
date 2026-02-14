"""Patchwise PSF estimation for spatially varying Fresnel blur calibration."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

LOGGER = logging.getLogger(__name__)


def estimate_psf_qp(
    blurred_patch: np.ndarray,
    sharp_patch: np.ndarray,
    psf_size: int = 31,
) -> np.ndarray:
    """Estimate a PSF using constrained quadratic programming.

    Implements the constrained least-squares kernel estimation described by
    Mannan and Langer (CalibPSF):

    ``min_p ||b - H p||^2`` subject to ``p_i >= 0`` and ``sum(p) = 1``.

    Parameters
    ----------
    blurred_patch : np.ndarray
        Blurred image patch.
    sharp_patch : np.ndarray
        Corresponding sharp image patch.
    psf_size : int, optional
        Square PSF support size. Must be an odd positive integer.

    Returns
    -------
    np.ndarray
        Estimated PSF of shape ``(psf_size, psf_size)`` and sum equal to one.
    """
    if psf_size <= 0 or psf_size % 2 == 0:
        raise ValueError("psf_size must be a positive odd integer.")

    blurred = _prepare_patch(blurred_patch)
    sharp = _prepare_patch(sharp_patch)
    if blurred.shape != sharp.shape:
        raise ValueError("blurred_patch and sharp_patch must have the same shape.")

    radius = psf_size // 2
    if min(blurred.shape) <= 2 * radius:
        raise ValueError("Patch is too small for the requested psf_size.")

    windows = _extract_sharp_windows(sharp, psf_size)
    blurred_valid = blurred[radius:-radius, radius:-radius]

    h_matrix = windows.reshape(-1, psf_size * psf_size)
    b_vector = blurred_valid.reshape(-1).astype(np.float64)

    max_rows = 5000
    if h_matrix.shape[0] > max_rows:
        idx = np.linspace(0, h_matrix.shape[0] - 1, num=max_rows, dtype=np.int64)
        h_matrix = h_matrix[idx]
        b_vector = b_vector[idx]

    k = psf_size * psf_size
    x0 = np.full(k, 1.0 / k, dtype=np.float64)

    def objective(psf_vec: np.ndarray) -> float:
        residual = b_vector - h_matrix @ psf_vec
        return float(residual @ residual)

    def jacobian(psf_vec: np.ndarray) -> np.ndarray:
        residual = b_vector - h_matrix @ psf_vec
        return -2.0 * (h_matrix.T @ residual)

    constraints = [{"type": "eq", "fun": lambda p: float(np.sum(p) - 1.0)}]
    bounds = [(0.0, None)] * k

    result = minimize(
        objective,
        x0,
        jac=jacobian,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )

    if not result.success:
        LOGGER.warning("SLSQP PSF estimation did not fully converge: %s", result.message)

    psf = np.asarray(result.x, dtype=np.float64).reshape(psf_size, psf_size)
    psf = np.maximum(psf, 0.0)
    total = float(psf.sum())
    if total <= 0.0:
        return np.full((psf_size, psf_size), 1.0 / (psf_size * psf_size), dtype=np.float64)
    return psf / total


def estimate_psf_wiener(
    blurred_patch: np.ndarray,
    sharp_patch: np.ndarray,
    noise_var: float = 0.01,
) -> np.ndarray:
    """Estimate a PSF from blurred/sharp patches using Wiener deconvolution.

    Parameters
    ----------
    blurred_patch : np.ndarray
        Blurred image patch.
    sharp_patch : np.ndarray
        Corresponding sharp image patch.
    noise_var : float, optional
        Small regularization value used in the frequency-domain division.

    Returns
    -------
    np.ndarray
        Estimated PSF cropped to a centered odd-sized support (up to 31x31).
    """
    blurred = _prepare_patch(blurred_patch)
    sharp = _prepare_patch(sharp_patch)
    if blurred.shape != sharp.shape:
        raise ValueError("blurred_patch and sharp_patch must have the same shape.")

    b_fft = np.fft.fft2(blurred)
    s_fft = np.fft.fft2(sharp)
    threshold = max(1e-8, float(noise_var) * 1e-3)
    safe_s_fft = np.where(np.abs(s_fft) < threshold, threshold + 0.0j, s_fft)

    psf_fft = b_fft / (safe_s_fft + float(noise_var))
    psf = np.real(np.fft.ifft2(psf_fft))
    psf = np.fft.fftshift(psf)

    psf = np.maximum(psf, 0.0)
    crop_size = min(31, psf.shape[0], psf.shape[1])
    if crop_size % 2 == 0:
        crop_size -= 1
    psf = _center_crop(psf, crop_size)
    psf_sum = float(psf.sum())
    if psf_sum <= 0.0:
        return np.full((crop_size, crop_size), 1.0 / (crop_size * crop_size), dtype=np.float64)
    return psf / psf_sum


def estimate_psf_rl(
    blurred_patch: np.ndarray,
    sharp_patch: np.ndarray,
    psf_size: int = 31,
    n_iter: int = 50,
) -> np.ndarray:
    """Estimate a PSF with Richardson-Lucy style iterative updates.

    Parameters
    ----------
    blurred_patch : np.ndarray
        Blurred image patch.
    sharp_patch : np.ndarray
        Corresponding sharp image patch.
    psf_size : int, optional
        Square PSF support size. Must be an odd positive integer.
    n_iter : int, optional
        Number of Richardson-Lucy iterations.

    Returns
    -------
    np.ndarray
        Estimated PSF of shape ``(psf_size, psf_size)`` and sum equal to one.
    """
    if psf_size <= 0 or psf_size % 2 == 0:
        raise ValueError("psf_size must be a positive odd integer.")
    if n_iter <= 0:
        raise ValueError("n_iter must be positive.")

    blurred = _prepare_patch(blurred_patch)
    sharp = _prepare_patch(sharp_patch)
    if blurred.shape != sharp.shape:
        raise ValueError("blurred_patch and sharp_patch must have the same shape.")

    psf = np.full((psf_size, psf_size), 1.0 / (psf_size * psf_size), dtype=np.float64)
    sharp_flipped = sharp[::-1, ::-1]
    eps = 1e-10
    sharp_sum = max(float(np.sum(sharp)), eps)

    for _ in range(n_iter):
        predicted = fftconvolve(sharp, psf, mode="same")
        ratio = blurred / (predicted + eps)
        correction_map = fftconvolve(ratio, sharp_flipped, mode="same")
        correction = _center_crop(correction_map, psf_size)

        psf = psf * correction / sharp_sum
        psf = np.maximum(psf, 0.0)
        total = float(psf.sum())
        if total <= eps:
            psf = np.full((psf_size, psf_size), 1.0 / (psf_size * psf_size), dtype=np.float64)
        else:
            psf = psf / total

    return psf


def estimate_spatially_varying_psf(
    blurred: np.ndarray,
    sharp: np.ndarray,
    patch_size: int = 64,
    psf_size: int = 31,
    overlap: float = 0.5,
    method: str = "qp",
    per_channel: bool = True,
) -> tuple[list[np.ndarray] | list[list[np.ndarray]], list[tuple[int, int]]]:
    """Estimate a patchwise spatially varying PSF field.

    Parameters
    ----------
    blurred : np.ndarray
        Blurred image, grayscale or color.
    sharp : np.ndarray
        Corresponding sharp image, same shape as ``blurred``.
    patch_size : int, optional
        Square patch size used for local estimation.
    psf_size : int, optional
        Square PSF support size used by QP/RL estimators.
    overlap : float, optional
        Patch overlap fraction in ``[0, 1)``.
    method : str, optional
        PSF estimator to use: ``"qp"``, ``"wiener"``, or ``"rl"``.
    per_channel : bool, optional
        When True and input has 3 channels, estimates separate PSF grids per channel.

    Returns
    -------
    tuple[list[np.ndarray] | list[list[np.ndarray]], list[tuple[int, int]]]
        ``(psf_grid, positions)`` where ``psf_grid`` contains one PSF per patch
        (or one list per color channel) and ``positions`` are patch centers as
        ``(row, col)``.
    """
    if blurred.shape != sharp.shape:
        raise ValueError("blurred and sharp must have the same shape.")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")

    from .alignment import align_patches, extract_patches

    if per_channel and blurred.ndim == 3 and blurred.shape[2] == 3:
        channel_psf_grid: list[list[np.ndarray]] = []
        shared_positions: list[tuple[int, int]] = []
        for c_idx in range(3):
            psf_grid, positions = _estimate_single_channel_grid(
                blurred[..., c_idx],
                sharp[..., c_idx],
                patch_size=patch_size,
                psf_size=psf_size,
                overlap=overlap,
                method=method,
            )
            channel_psf_grid.append(psf_grid)
            if not shared_positions:
                shared_positions = positions
        return channel_psf_grid, shared_positions

    psf_grid, positions = _estimate_single_channel_grid(
        blurred,
        sharp,
        patch_size=patch_size,
        psf_size=psf_size,
        overlap=overlap,
        method=method,
    )
    return psf_grid, positions


def interpolate_psf(
    psf_grid: Sequence[np.ndarray],
    positions: Sequence[tuple[int, int]],
    query_point: tuple[float, float],
) -> np.ndarray:
    """Interpolate PSF at an arbitrary position using bilinear interpolation.

    Parameters
    ----------
    psf_grid : Sequence[np.ndarray]
        Sequence of PSFs associated with ``positions``.
    positions : Sequence[tuple[int, int]]
        Patch-center coordinates as ``(row, col)``.
    query_point : tuple[float, float]
        Query coordinate ``(row, col)`` where the PSF should be interpolated.

    Returns
    -------
    np.ndarray
        Interpolated PSF. If query is outside known bounds, nearest PSF is used.
    """
    if len(psf_grid) == 0 or len(positions) == 0:
        raise ValueError("psf_grid and positions must be non-empty.")
    if len(psf_grid) != len(positions):
        raise ValueError("psf_grid and positions must have matching lengths.")

    row_q = float(query_point[0])
    col_q = float(query_point[1])

    rows = np.array([pos[0] for pos in positions], dtype=np.float64)
    cols = np.array([pos[1] for pos in positions], dtype=np.float64)

    row_min, row_max = float(rows.min()), float(rows.max())
    col_min, col_max = float(cols.min()), float(cols.max())
    if row_q < row_min or row_q > row_max or col_q < col_min or col_q > col_max:
        return _nearest_psf(psf_grid, positions, query_point)

    unique_rows = np.unique(rows)
    unique_cols = np.unique(cols)

    r0, r1 = _bounding_values(unique_rows, row_q)
    c0, c1 = _bounding_values(unique_cols, col_q)

    psf_lookup = {tuple(pos): psf for pos, psf in zip(positions, psf_grid, strict=True)}
    corners = [(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))]
    if any(corner not in psf_lookup for corner in corners):
        return _nearest_psf(psf_grid, positions, query_point)

    if r0 == r1 and c0 == c1:
        return psf_lookup[(int(r0), int(c0))].copy()
    if r0 == r1:
        t = 0.0 if c1 == c0 else (col_q - c0) / (c1 - c0)
        psf = (1.0 - t) * psf_lookup[(int(r0), int(c0))] + t * psf_lookup[(int(r0), int(c1))]
        return _normalize_psf(psf)
    if c0 == c1:
        t = 0.0 if r1 == r0 else (row_q - r0) / (r1 - r0)
        psf = (1.0 - t) * psf_lookup[(int(r0), int(c0))] + t * psf_lookup[(int(r1), int(c0))]
        return _normalize_psf(psf)

    wr = (row_q - r0) / (r1 - r0)
    wc = (col_q - c0) / (c1 - c0)

    top = (1.0 - wc) * psf_lookup[(int(r0), int(c0))] + wc * psf_lookup[(int(r0), int(c1))]
    bottom = (1.0 - wc) * psf_lookup[(int(r1), int(c0))] + wc * psf_lookup[(int(r1), int(c1))]
    interp = (1.0 - wr) * top + wr * bottom
    return _normalize_psf(interp)


def _estimate_single_channel_grid(
    blurred: np.ndarray,
    sharp: np.ndarray,
    patch_size: int,
    psf_size: int,
    overlap: float,
    method: str,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Align, patch, and estimate PSFs for one grayscale channel."""
    from .alignment import align_patches, extract_patches

    aligned_pairs = align_patches(blurred, sharp, patch_size=patch_size, overlap=overlap)

    estimator = method.lower()
    psf_grid: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

    for blurred_patch, sharp_patch, origin in aligned_pairs:
        centre = (
            origin[0] + blurred_patch.shape[0] // 2,
            origin[1] + blurred_patch.shape[1] // 2,
        )
        if estimator == "qp":
            psf = estimate_psf_qp(blurred_patch, sharp_patch, psf_size=psf_size)
        elif estimator == "wiener":
            wiener_psf = estimate_psf_wiener(blurred_patch, sharp_patch)
            psf = _normalize_psf(_fit_to_size(wiener_psf, psf_size))
        elif estimator == "rl":
            psf = estimate_psf_rl(blurred_patch, sharp_patch, psf_size=psf_size)
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'qp', 'wiener', or 'rl'.")
        psf_grid.append(psf)
        positions.append(centre)

    if not psf_grid:
        patches, origins = extract_patches(blurred, patch_size, overlap)
        sharp_patches_list, _ = extract_patches(sharp, patch_size, overlap)
        n = min(len(patches), len(sharp_patches_list))
        for i in range(n):
            centre = (
                origins[i][0] + patches[i].shape[0] // 2,
                origins[i][1] + patches[i].shape[1] // 2,
            )
            if estimator == "qp":
                psf = estimate_psf_qp(patches[i], sharp_patches_list[i], psf_size=psf_size)
            elif estimator == "wiener":
                wiener_psf = estimate_psf_wiener(patches[i], sharp_patches_list[i])
                psf = _normalize_psf(_fit_to_size(wiener_psf, psf_size))
            elif estimator == "rl":
                psf = estimate_psf_rl(patches[i], sharp_patches_list[i], psf_size=psf_size)
            else:
                raise ValueError(
                    f"Unsupported method '{method}'. Use 'qp', 'wiener', or 'rl'."
                )
            psf_grid.append(psf)
            positions.append(centre)

    return psf_grid, positions


def _prepare_patch(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch.astype(np.float32), cv2.COLOR_BGR2GRAY)
    if patch.ndim != 2:
        raise ValueError("Patch must be 2D or 3-channel color.")

    patch_f = np.asarray(patch, dtype=np.float64)
    min_v = float(np.min(patch_f))
    max_v = float(np.max(patch_f))
    if min_v < 0.0 or max_v > 1.0:
        if np.issubdtype(patch.dtype, np.integer):
            info = np.iinfo(patch.dtype)
            patch_f = (patch_f - info.min) / float(info.max - info.min)
        elif max_v > min_v:
            patch_f = (patch_f - min_v) / (max_v - min_v)
        else:
            patch_f = np.zeros_like(patch_f)
    return np.clip(patch_f, 0.0, 1.0)


def _extract_sharp_windows(sharp: np.ndarray, psf_size: int) -> np.ndarray:
    radius = psf_size // 2
    sharp_padded = np.pad(sharp, ((radius, radius), (radius, radius)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(sharp_padded, (psf_size, psf_size))
    return windows[radius:-radius, radius:-radius]


def _center_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    crop_size = min(crop_size, array.shape[0], array.shape[1])
    if crop_size % 2 == 0:
        crop_size -= 1
    center_r = array.shape[0] // 2
    center_c = array.shape[1] // 2
    half = crop_size // 2
    r0 = center_r - half
    r1 = center_r + half + 1
    c0 = center_c - half
    c1 = center_c + half + 1
    return np.asarray(array[r0:r1, c0:c1], dtype=np.float64)


def _fit_to_size(psf: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0 or target_size % 2 == 0:
        raise ValueError("target_size must be a positive odd integer.")
    if psf.shape == (target_size, target_size):
        return psf
    if psf.shape[0] >= target_size and psf.shape[1] >= target_size:
        return _center_crop(psf, target_size)

    out = np.zeros((target_size, target_size), dtype=np.float64)
    source = psf
    if source.shape[0] > target_size or source.shape[1] > target_size:
        source = _center_crop(source, target_size)

    r0 = (target_size - source.shape[0]) // 2
    c0 = (target_size - source.shape[1]) // 2
    out[r0 : r0 + source.shape[0], c0 : c0 + source.shape[1]] = source
    return out


def _normalize_psf(psf: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(psf, dtype=np.float64), 0.0)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(clipped.shape, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


def _nearest_psf(
    psf_grid: Sequence[np.ndarray],
    positions: Sequence[tuple[int, int]],
    query_point: tuple[float, float],
) -> np.ndarray:
    row_q, col_q = query_point
    distances = [
        (float((row_q - r) ** 2 + (col_q - c) ** 2), idx)
        for idx, (r, c) in enumerate(positions)
    ]
    nearest_idx = min(distances, key=lambda item: item[0])[1]
    return np.asarray(psf_grid[nearest_idx], dtype=np.float64).copy()


def _bounding_values(values: np.ndarray, query: float) -> tuple[float, float]:
    lower = values[values <= query]
    upper = values[values >= query]
    v0 = float(lower[-1]) if lower.size > 0 else float(values[0])
    v1 = float(upper[0]) if upper.size > 0 else float(values[-1])
    return v0, v1



