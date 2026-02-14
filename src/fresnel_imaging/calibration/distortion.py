"""Lens distortion calibration and correction for Fresnel imaging.

Fresnel lenses can introduce strong geometric distortions and groove artifacts
that frequently break standard checkerboard corner detection. This module
implements robust corner extraction with fallback pre-processing and OpenCV
calibration helpers for single-image and multi-image workflows.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class CalibrationResult(NamedTuple):
    camera_matrix: NDArray[np.floating]
    dist_coeffs: NDArray[np.floating]
    rvecs: list[NDArray[np.floating]]
    tvecs: list[NDArray[np.floating]]
    rms_error: float


def estimate_distortion(
    blurred_image: np.ndarray,
    pattern_size: tuple[int, int] = (8, 6),
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    """Estimate distortion parameters from one checkerboard image.

    Parameters
    ----------
    blurred_image : np.ndarray
        Distorted checkerboard image. Can be grayscale or color.
    pattern_size : tuple[int, int], optional
        Number of inner checkerboard corners as ``(columns, rows)``.

    Returns
    -------
    tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]
        ``(camera_matrix, dist_coeffs)`` if corners are detected,
        otherwise ``(None, None)``.
    """
    gray = _prepare_grayscale_uint8(blurred_image)
    corners = _detect_checkerboard_corners(gray, pattern_size)
    if corners is None:
        LOGGER.warning(
            "Checkerboard detection failed for distortion estimation; "
            "returning no calibration."
        )
        return None, None

    object_points = [_build_object_points(pattern_size, square_size=1.0)]
    image_points = [corners]
    image_size = (gray.shape[1], gray.shape[0])

    try:
        _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    except cv2.error as exc:
        LOGGER.warning("OpenCV calibration failed in estimate_distortion: %s", exc)
        return None, None

    return camera_matrix, dist_coeffs


def undistort_image(
    image: np.ndarray,
    camera_matrix: NDArray[np.floating],
    dist_coeffs: NDArray[np.floating],
) -> np.ndarray:
    """Remove lens distortion from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or multi-channel).
    camera_matrix : NDArray[np.floating]
        Camera intrinsic matrix.
    dist_coeffs : NDArray[np.floating]
        Distortion coefficients.

    Returns
    -------
    np.ndarray
        Undistorted image.
    """
    height, width = image.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        (width, height),
        1.0,
        (width, height),
    )
    return cv2.undistort(
        image,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        None,
        new_camera_matrix,
    )


def calibrate_from_multiple(
    images: Sequence[np.ndarray],
    pattern_size: tuple[int, int] = (8, 6),
    square_size: float = 25.0,
    manual_corners: Sequence[NDArray[np.floating] | np.ndarray | None] | None = None,
) -> CalibrationResult | None:
    """Calibrate a camera from multiple checkerboard images.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Sequence of checkerboard views.
    pattern_size : tuple[int, int], optional
        Number of inner checkerboard corners as ``(columns, rows)``.
    square_size : float, optional
        Physical checkerboard square size in world units.
    manual_corners : Sequence[NDArray[np.floating] | np.ndarray | None] | None, optional
        Optional pre-detected corner arrays aligned with ``images`` indices.
        When provided and non-None for a given image, these corners are used
        instead of automatic detection.

    Returns
    -------
    CalibrationResult | None
        Calibration output container, or ``None`` if not enough valid corner
        detections are available.
    """
    if len(images) == 0:
        LOGGER.warning("No images were provided for calibration.")
        return None

    template_obj_points = _build_object_points(pattern_size, square_size=square_size)
    object_points: list[NDArray[np.floating]] = []
    image_points: list[NDArray[np.floating]] = []
    image_size: tuple[int, int] | None = None

    for idx, image in enumerate(images):
        gray = _prepare_grayscale_uint8(image)
        image_size = (gray.shape[1], gray.shape[0])

        selected_corners: NDArray[np.floating] | None = None
        if manual_corners is not None and idx < len(manual_corners):
            manual = manual_corners[idx]
            if manual is not None:
                try:
                    selected_corners = _format_corner_array(manual, pattern_size)
                    LOGGER.info("Using manual corners for image %d.", idx)
                except ValueError as exc:
                    LOGGER.warning(
                        "Invalid manual corners for image %d (%s); retrying auto-detection.",
                        idx,
                        exc,
                    )

        if selected_corners is None:
            selected_corners = _detect_checkerboard_corners(gray, pattern_size)

        if selected_corners is None:
            LOGGER.warning("Skipping image %d: checkerboard corners not found.", idx)
            continue

        object_points.append(template_obj_points.copy())
        image_points.append(selected_corners)

    if not image_points or image_size is None:
        LOGGER.warning("Calibration failed: no valid checkerboard detections.")
        return None

    try:
        rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    except cv2.error as exc:
        LOGGER.warning("OpenCV calibration failed in calibrate_from_multiple: %s", exc)
        return None

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        rms_error=float(rms_error),
    )


def _detect_checkerboard_corners(
    gray: NDArray[np.uint8],
    pattern_size: tuple[int, int],
) -> NDArray[np.floating] | None:
    base_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, base_flags)

    if not found:
        LOGGER.debug("Initial checkerboard detection failed, using preprocessing fallback.")
        preprocessed = _fresnel_preprocess(gray)
        found, corners = cv2.findChessboardCorners(preprocessed, pattern_size, base_flags)
        if not found:
            inverted = cv2.bitwise_not(preprocessed)
            found, corners = cv2.findChessboardCorners(inverted, pattern_size, base_flags)

    if not found or corners is None:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return np.asarray(refined, dtype=np.float32)


def _fresnel_preprocess(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


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


def _build_object_points(
    pattern_size: tuple[int, int],
    square_size: float,
) -> NDArray[np.floating]:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size)
    return objp


def _format_corner_array(
    corners: NDArray[np.floating] | np.ndarray,
    pattern_size: tuple[int, int],
) -> NDArray[np.floating]:
    expected = pattern_size[0] * pattern_size[1]
    corners_arr = np.asarray(corners, dtype=np.float32)

    if corners_arr.ndim == 2 and corners_arr.shape == (expected, 2):
        corners_arr = corners_arr[:, np.newaxis, :]
    elif corners_arr.ndim == 3 and corners_arr.shape == (expected, 1, 2):
        pass
    else:
        raise ValueError(
            f"Corners must have shape ({expected}, 2) or ({expected}, 1, 2), "
            f"got {corners_arr.shape}."
        )

    return corners_arr
