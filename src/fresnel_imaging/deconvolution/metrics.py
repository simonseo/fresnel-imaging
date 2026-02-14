"""Image quality metrics for deconvolution evaluation.

Provides SNR, PSNR, and SSIM metrics for comparing deconvolved images
against reference (ground-truth) images.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity


def snr(
    sig: NDArray[np.floating],
    hk: int,
    ref: NDArray[np.floating],
) -> tuple[float, float]:
    """Compute Signal-to-Noise Ratio for images.

    Port of snr.m. Uses the central portion of the image, cropping
    ``hk * 4`` pixels from each edge to avoid border artefacts.

    Parameters
    ----------
    sig : ndarray
        Modified / deconvolved image.
    hk : int
        Half-kernel size. The crop margin is ``hk * 4``.
    ref : ndarray
        Reference (ground-truth) image.

    Returns
    -------
    snr_db : float
        SNR in decibels: ``10 * log10(variance(ref_crop) / MSE)``.
    mse_value : float
        Mean squared error over the cropped region.
    """
    margin = hk * 4

    # Crop central portion (works for 2-D and 3-D images)
    if ref.ndim == 2:
        ref_crop = ref[margin:-margin, margin:-margin]
        sig_crop = sig[margin:-margin, margin:-margin]
    else:
        ref_crop = ref[margin:-margin, margin:-margin, :]
        sig_crop = sig[margin:-margin, margin:-margin, :]

    mse_value = float(np.mean((ref_crop.ravel() - sig_crop.ravel()) ** 2))

    # Biased variance (MATLAB's var with weight flag 1)
    dv = float(np.var(ref_crop.ravel(), ddof=0))

    snr_db = 10.0 * np.log10(dv / mse_value) if mse_value > 0 else float("inf")

    return float(snr_db), mse_value


def psnr(
    img: NDArray[np.floating],
    ref: NDArray[np.floating],
    crop_pad: int = 0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    img : ndarray
        Deconvolved image (values assumed in [0, 1]).
    ref : ndarray
        Reference (ground-truth) image.
    crop_pad : int, optional
        Number of pixels to crop from each edge before computing PSNR.
        Default is 0 (no cropping).

    Returns
    -------
    psnr_db : float
        PSNR in decibels. Returns ``inf`` if MSE is essentially zero.
    """
    if crop_pad > 0:
        slc = slice(crop_pad, -crop_pad)
        if img.ndim == 2:
            img = img[slc, slc]
            ref = ref[slc, slc]
        else:
            img = img[slc, slc, :]
            ref = ref[slc, slc, :]

    diff = ref.ravel() - img.ravel()
    mse = float(np.dot(diff, diff)) / diff.size

    if mse < np.finfo(float).eps:
        return float("inf")

    return float(10.0 * np.log10(1.0 / mse))


def ssim(
    img: NDArray[np.floating],
    ref: NDArray[np.floating],
) -> float:
    """Compute Structural Similarity Index Measure.

    Parameters
    ----------
    img : ndarray
        Deconvolved image.
    ref : ndarray
        Reference image.

    Returns
    -------
    ssim_value : float
        SSIM in [0, 1] (higher is better).
    """
    # Determine if multichannel
    multichannel = img.ndim == 3 and img.shape[2] > 1
    return float(
        structural_similarity(
            ref,
            img,
            data_range=1.0,
            channel_axis=2 if multichannel else None,
        )
    )
