"""Image alignment utilities for PSF estimation in Fresnel imaging."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


def align_global(
    blurred: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, NDArray[np.float64]]:
    """Globally align a blurred image to a reference using SIFT and homography.

    Parameters
    ----------
    blurred : np.ndarray
        Input blurred image, either grayscale ``(H, W)`` or color ``(H, W, C)``.
    reference : np.ndarray
        Target reference image, either grayscale ``(H, W)`` or color ``(H, W, C)``.

    Returns
    -------
    tuple[np.ndarray, NDArray[np.float64]]
        Aligned blurred image warped into reference coordinates, and the estimated
        ``3 x 3`` homography matrix. If alignment fails, returns the original
        blurred image and an identity transform.
    """
    homography = _estimate_homography_sift_ransac(blurred, reference)
    if homography is None:
        LOGGER.warning(
            "Global alignment failed: insufficient or unreliable matches; using identity transform."
        )
        return blurred.copy(), np.eye(3, dtype=np.float64)

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(blurred, homography, (width, height))
    return aligned, homography


def align_patches(
    blurred: np.ndarray,
    reference: np.ndarray,
    patch_size: int = 64,
    overlap: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    """Perform local patch-wise alignment after coarse global registration.

    Parameters
    ----------
    blurred : np.ndarray
        Input blurred image to align.
    reference : np.ndarray
        Reference image in target coordinates.
    patch_size : int, optional
        Square patch size in pixels.
    overlap : float, optional
        Fractional overlap in ``[0, 1)`` between adjacent patches.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, tuple[int, int]]]
        List of tuples ``(aligned_blurred_patch, reference_patch, origin)`` where
        ``origin`` is ``(row_start, col_start)``. If local alignment fails for a
        patch, the globally aligned patch is returned for that location.
    """
    globally_aligned, _ = align_global(blurred, reference)

    blurred_patches, blurred_origins = extract_patches(globally_aligned, patch_size, overlap)
    reference_patches, reference_origins = extract_patches(reference, patch_size, overlap)

    if blurred_origins != reference_origins:
        LOGGER.warning(
            "Patch grids differ between blurred and reference images; "
            "using reference grid for pair construction."
        )

    aligned_patch_pairs: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
    patch_count = min(len(blurred_patches), len(reference_patches))
    for idx in range(patch_count):
        blurred_patch = blurred_patches[idx]
        reference_patch = reference_patches[idx]
        origin = reference_origins[idx]

        local_homography = _estimate_homography_sift_ransac(blurred_patch, reference_patch)
        if local_homography is None:
            aligned_patch = blurred_patch
        else:
            patch_height, patch_width = reference_patch.shape[:2]
            aligned_patch = cv2.warpPerspective(
                blurred_patch,
                local_homography,
                (patch_width, patch_height),
            )

        aligned_patch_pairs.append((aligned_patch, reference_patch, origin))

    return aligned_patch_pairs


def extract_patches(
    image: np.ndarray,
    patch_size: int,
    overlap: float,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Extract overlapping square patches and their origins from an image.

    Parameters
    ----------
    image : np.ndarray
        Source image, grayscale ``(H, W)`` or color ``(H, W, C)``.
    patch_size : int
        Nominal size of square patches in pixels.
    overlap : float
        Fractional overlap in ``[0, 1)`` between adjacent patches.

    Returns
    -------
    tuple[list[np.ndarray], list[tuple[int, int]]]
        Extracted patches and corresponding top-left origins ``(row, col)``.
        Edge patches are included and may be smaller than ``patch_size``.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1, got {overlap}.")

    stride = max(int(patch_size * (1.0 - overlap)), 1)
    height, width = image.shape[:2]

    row_starts = _compute_patch_starts(height, patch_size, stride)
    col_starts = _compute_patch_starts(width, patch_size, stride)

    patches: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for row_start in row_starts:
        row_end = min(row_start + patch_size, height)
        for col_start in col_starts:
            col_end = min(col_start + patch_size, width)
            patches.append(image[row_start:row_end, col_start:col_end].copy())
            origins.append((row_start, col_start))

    return patches, origins


def stitch_patches(
    patches: list[np.ndarray],
    origins: list[tuple[int, int]],
    image_shape: tuple[int, ...],
    overlap: float,
) -> np.ndarray:
    """Stitch patches into an image with feathered blending in overlap regions.

    Parameters
    ----------
    patches : list[np.ndarray]
        Patch list to composite back into an image.
    origins : list[tuple[int, int]]
        Patch top-left coordinates as ``(row_start, col_start)``.
    image_shape : tuple[int, ...]
        Output image shape, either ``(H, W)`` or ``(H, W, C)``.
    overlap : float
        Fractional overlap in ``[0, 1)`` used during extraction.

    Returns
    -------
    np.ndarray
        Stitched image with weighted blending across overlaps.
    """
    if len(patches) != len(origins):
        raise ValueError("patches and origins must have the same length.")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1, got {overlap}.")
    if not patches:
        return np.zeros(image_shape, dtype=np.float32)

    if len(image_shape) not in {2, 3}:
        raise ValueError(f"image_shape must have 2 or 3 dimensions, got {image_shape}.")

    is_color = len(image_shape) == 3
    output = np.zeros(image_shape, dtype=np.float32)
    weights = np.zeros(image_shape[:2], dtype=np.float32)

    image_height, image_width = image_shape[:2]
    for patch, (row_start, col_start) in zip(patches, origins):
        if patch.ndim not in {2, 3}:
            raise ValueError(f"Each patch must be 2D or 3D, got shape {patch.shape}.")

        patch_height, patch_width = patch.shape[:2]
        row_end = min(row_start + patch_height, image_height)
        col_end = min(col_start + patch_width, image_width)
        if row_end <= row_start or col_end <= col_start:
            continue

        valid_height = row_end - row_start
        valid_width = col_end - col_start
        patch_view = patch[:valid_height, :valid_width]

        weight = _feather_mask(valid_height, valid_width, overlap)
        patch_float = np.asarray(patch_view, dtype=np.float32)

        if is_color:
            output[row_start:row_end, col_start:col_end] += patch_float * weight[..., np.newaxis]
        else:
            output[row_start:row_end, col_start:col_end] += patch_float * weight

        weights[row_start:row_end, col_start:col_end] += weight

    eps = np.finfo(np.float32).eps
    if is_color:
        stitched = output / np.maximum(weights[..., np.newaxis], eps)
    else:
        stitched = output / np.maximum(weights, eps)

    target_dtype = np.asarray(patches[0]).dtype
    if np.issubdtype(target_dtype, np.integer):
        dtype_info = np.iinfo(target_dtype)
        stitched = np.clip(np.rint(stitched), dtype_info.min, dtype_info.max)
    return stitched.astype(target_dtype, copy=False)


def _estimate_homography_sift_ransac(
    source: np.ndarray,
    target: np.ndarray,
) -> NDArray[np.float64] | None:
    source_gray = _prepare_grayscale_uint8(source)
    target_gray = _prepare_grayscale_uint8(target)

    try:
        sift = cv2.SIFT_create()
        source_keypoints, source_desc = sift.detectAndCompute(source_gray, None)
        target_keypoints, target_desc = sift.detectAndCompute(target_gray, None)
    except cv2.error as exc:
        LOGGER.warning("SIFT feature extraction failed: %s", exc)
        return None

    if source_desc is None or target_desc is None:
        return None
    if len(source_keypoints) < 4 or len(target_keypoints) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(source_desc, target_desc, k=2)

    good_matches: list[cv2.DMatch] = []
    ratio = 0.75
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good_matches.append(best)

    if len(good_matches) < 4:
        return None

    source_points = np.float32(
        [source_keypoints[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [target_keypoints[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if homography is None:
        return None
    return np.asarray(homography, dtype=np.float64)


def _prepare_grayscale_uint8(image: np.ndarray) -> NDArray[np.uint8]:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    if gray.dtype == np.uint8:
        return gray

    gray_float = np.asarray(gray, dtype=np.float32)
    gray_norm = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
    return np.asarray(gray_norm, dtype=np.uint8)


def _compute_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]

    starts = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _feather_mask(height: int, width: int, overlap: float) -> NDArray[np.float32]:
    y = np.ones(height, dtype=np.float32)
    x = np.ones(width, dtype=np.float32)

    overlap_y = min(int(round(height * overlap)), max(height // 2, 1))
    overlap_x = min(int(round(width * overlap)), max(width // 2, 1))

    if overlap_y > 0 and height > 1:
        ramp_y = np.linspace(0.0, 1.0, overlap_y + 2, dtype=np.float32)[1:-1]
        y[:overlap_y] = np.minimum(y[:overlap_y], ramp_y)
        y[-overlap_y:] = np.minimum(y[-overlap_y:], ramp_y[::-1])

    if overlap_x > 0 and width > 1:
        ramp_x = np.linspace(0.0, 1.0, overlap_x + 2, dtype=np.float32)[1:-1]
        x[:overlap_x] = np.minimum(x[:overlap_x], ramp_x)
        x[-overlap_x:] = np.minimum(x[-overlap_x:], ramp_x[::-1])

    return np.outer(y, x)
