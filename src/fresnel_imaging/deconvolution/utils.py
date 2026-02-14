"""Utility functions for deconvolution algorithms.

Ported from MATLAB code by Felix Heide (fheide@cs.ubc.ca).
Part of the computational imaging pipeline from:
F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
High-Quality Computational Imaging Through Simple Lenses. ACM ToG 2013.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.signal
from numpy.fft import fft2


def psf2otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert a point-spread function (PSF) to an optical transfer function (OTF).

    Matches MATLAB's ``psf2otf``: zero-pads *psf* to *shape*, circularly shifts
    the center of the PSF to the origin (top-left corner), then computes the 2-D
    FFT.

    Parameters
    ----------
    psf : np.ndarray
        2-D point-spread function (kernel).
    shape : tuple[int, int]
        Desired output shape ``(rows, cols)`` — typically the image size.

    Returns
    -------
    np.ndarray
        Complex-valued OTF of shape *shape*.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=complex)

    # Pad PSF to desired shape
    pad_shape = np.array(shape) - np.array(psf.shape)
    psf_padded = np.pad(psf, [(0, pad_shape[0]), (0, pad_shape[1])])

    # Circularly shift so that the center of the PSF moves to (0, 0).
    # MATLAB circshift uses -floor(size(psf)/2).
    shift = -(np.array(psf.shape) // 2)
    psf_padded = np.roll(psf_padded, shift, axis=(0, 1))

    return fft2(psf_padded)


def edgetaper(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Taper image edges using PSF autocorrelation to reduce ringing.

    Matches MATLAB's ``edgetaper``.  Computes the normalised autocorrelation of
    the PSF as an alpha mask (1 in the centre, tapering to 0 at edges), then
    blends::

        output = image * alpha + blurred_image * (1 - alpha)

    where ``blurred_image = conv(image, psf)`` (same size, zero-padded FFT
    convolution).

    Parameters
    ----------
    image : np.ndarray
        Input 2-D image.
    psf : np.ndarray
        Point-spread function used for tapering.

    Returns
    -------
    np.ndarray
        Edge-tapered image, same shape as *image*.
    """
    # Compute autocorrelation of the PSF: xcorr2(psf, psf)
    # In MATLAB: xcorr2(psf, psf) == conv2(psf, rot180(psf), 'full')
    psf_flipped = psf[::-1, ::-1]
    autocorr = scipy.signal.fftconvolve(psf, psf_flipped, mode="full")

    # Normalise to [0, 1]
    autocorr = autocorr / autocorr.max()

    # Build separable alpha mask for the image.
    # MATLAB edgetaper builds 1-D tapers from the autocorrelation's marginals.
    # Vertical taper from summing autocorrelation along columns (axis=1),
    # horizontal taper from summing along rows (axis=0).
    ac_rows = autocorr.shape[0]
    ac_cols = autocorr.shape[1]
    img_rows, img_cols = image.shape[:2]

    # 1-D vertical taper: take center column of autocorrelation
    beta_v = autocorr[:, ac_cols // 2]
    beta_v = beta_v / beta_v.max()

    # 1-D horizontal taper: take center row of autocorrelation
    beta_h = autocorr[ac_rows // 2, :]
    beta_h = beta_h / beta_h.max()

    # Build alpha vectors for the image dimensions.
    # The taper is only applied near edges; the center is 1.
    half_ac_r = ac_rows // 2
    half_ac_c = ac_cols // 2

    alpha_v = np.ones(img_rows)
    # Top edge
    top_len = min(half_ac_r, img_rows)
    alpha_v[:top_len] = beta_v[half_ac_r:half_ac_r + top_len]
    # Bottom edge
    bot_len = min(half_ac_r, img_rows)
    alpha_v[-bot_len:] = np.minimum(
        alpha_v[-bot_len:], beta_v[half_ac_r - bot_len + 1 : half_ac_r + 1][::-1]
    )

    alpha_h = np.ones(img_cols)
    # Left edge
    left_len = min(half_ac_c, img_cols)
    alpha_h[:left_len] = beta_h[half_ac_c:half_ac_c + left_len]
    # Right edge
    right_len = min(half_ac_c, img_cols)
    alpha_h[-right_len:] = np.minimum(
        alpha_h[-right_len:], beta_h[half_ac_c - right_len + 1 : half_ac_c + 1][::-1]
    )

    # 2-D alpha mask via outer product
    alpha = alpha_v[:, np.newaxis] * alpha_h[np.newaxis, :]

    # Compute blurred image using FFT convolution (same size, zero-padded)
    blurred = scipy.signal.fftconvolve(image, psf, mode="same")

    # Blend
    return image * alpha + blurred * (1.0 - alpha)


def imconv(F: np.ndarray, K: np.ndarray, output: str = "same") -> np.ndarray:
    """Convolution with replicate boundary conditions.

    Matches the MATLAB ``imconv`` helper in ``pd_joint_deconv.m``.

    For small 2-element kernels (1×2 or 2×1) with ``output='full'``, uses the
    fast shift trick from the MATLAB code (replicate first/last row or column).
    For ``'same'`` output, uses ``scipy.ndimage.convolve(mode='nearest')``.
    For larger kernels with ``'full'`` output, uses FFT-based convolution with
    :func:`boundary_transform_deblurring`.

    Parameters
    ----------
    F : np.ndarray
        Input 2-D (or 3-D) array.
    K : np.ndarray
        Convolution kernel.
    output : str
        ``'same'`` or ``'full'``.  ``'same'`` returns the same size as *F*;
        ``'full'`` returns the full convolution result.

    Returns
    -------
    np.ndarray
        Filtered array.
    """
    if output == "full":
        # Fast path for small 2-element kernels
        if K.shape == (1, 2):
            # MATLAB: K(1,2)*F(:,[1 1:end],:) + K(1,1)*F(:,[1:end end],:)
            # Replicate first column on the left, last column on the right
            F_left = np.concatenate([F[:, :1], F], axis=1)   # F(:,[1 1:end])
            F_right = np.concatenate([F, F[:, -1:]], axis=1)  # F(:,[1:end end])
            return K[0, 1] * F_left + K[0, 0] * F_right
        elif K.shape == (2, 1):
            # MATLAB: K(2,1)*F([1 1:end],:,:) + K(1,1)*F([1:end end],:,:)
            F_top = np.concatenate([F[:1, :], F], axis=0)     # F([1 1:end])
            F_bot = np.concatenate([F, F[-1:, :]], axis=0)    # F([1:end end])
            return K[1, 0] * F_top + K[0, 0] * F_bot
        else:
            # General full convolution with replicate boundary.
            # Determine boundary padding size from kernel.
            pad_r = K.shape[0] - 1
            pad_c = K.shape[1] - 1
            if K.size > 25:
                # FFT-based for large kernels
                F_padded = boundary_transform_deblurring(
                    F, "add", max(pad_r, pad_c)
                )
                result = scipy.ndimage.convolve(F_padded, K, mode="nearest")
                # For 'full' output the result should have size = F + K - 1.
                # We padded by max(pad_r,pad_c) on each side, which is at least
                # K-1 on each side. Crop to the correct full size.
                out_r = F.shape[0] + K.shape[0] - 1
                out_c = F.shape[1] + K.shape[1] - 1
                # Center the crop
                start_r = (result.shape[0] - out_r) // 2
                start_c = (result.shape[1] - out_c) // 2
                return result[start_r : start_r + out_r, start_c : start_c + out_c]
            else:
                # For small-to-medium kernels, pad and use ndimage
                # Pad F with replicate boundary by (K-1) on each side
                if F.ndim == 2:
                    F_padded = np.pad(
                        F, [(pad_r, pad_r), (pad_c, pad_c)], mode="edge"
                    )
                else:
                    F_padded = np.pad(
                        F,
                        [(pad_r, pad_r), (pad_c, pad_c), (0, 0)],
                        mode="edge",
                    )
                # Apply convolution in 'same' mode on the padded array
                if F_padded.ndim == 3:
                    result = np.stack(
                        [
                            scipy.ndimage.convolve(
                                F_padded[:, :, c], K, mode="nearest"
                            )
                            for c in range(F_padded.shape[2])
                        ],
                        axis=2,
                    )
                else:
                    result = scipy.ndimage.convolve(
                        F_padded, K, mode="nearest"
                    )
                # Crop to full convolution size: input_size + kernel_size - 1
                out_r = F.shape[0] + K.shape[0] - 1
                out_c = F.shape[1] + K.shape[1] - 1
                # The padded result is (F + 2*(K-1)), we want (F + K - 1),
                # so crop (K-1)/2 from each side.
                start_r = (result.shape[0] - out_r) // 2
                start_c = (result.shape[1] - out_c) // 2
                return result[start_r : start_r + out_r, start_c : start_c + out_c]
    else:
        # 'same' output — use scipy.ndimage.convolve with replicate boundary
        # MATLAB imfilter('conv','replicate') matches mode='nearest'
        if F.ndim == 3:
            return np.stack(
                [
                    scipy.ndimage.convolve(F[:, :, c], K, mode="nearest")
                    for c in range(F.shape[2])
                ],
                axis=2,
            )
        return scipy.ndimage.convolve(F, K, mode="nearest")


def img_to_norm_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to a single-channel float array in [0, 1].

    Matches MATLAB's ``img_to_norm_grayscale``.  If the input has multiple
    channels it is first converted to grayscale via luminance weights.  Float
    inputs are normalised with ``(I - min) / (max - min)``; integer inputs are
    scaled by the datatype range.

    Parameters
    ----------
    img : np.ndarray
        Input image — 2-D (grayscale) or 3-D (colour, channels last).

    Returns
    -------
    np.ndarray
        2-D float64 array in [0, 1].
    """
    # Convert to single channel if multichannel
    if img.ndim == 3 and img.shape[2] > 1:
        # Use ITU-R BT.601 luma weights (same as MATLAB rgb2gray)
        img = (
            0.2989 * img[:, :, 0]
            + 0.5870 * img[:, :, 1]
            + 0.1140 * img[:, :, 2]
        )

    img = np.squeeze(img)

    if np.issubdtype(img.dtype, np.floating):
        # mat2gray equivalent: normalise to [0, 1] using min/max
        imin = img.min()
        imax = img.max()
        if imax - imin == 0:
            return np.zeros_like(img, dtype=np.float64)
        return (img.astype(np.float64) - imin) / (imax - imin)
    elif np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return (img.astype(np.float64) - info.min) / (info.max - info.min)
    else:
        raise TypeError(
            f"Normalization could not be performed for dtype {img.dtype}"
        )


def img_mult(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Multiply each pixel of a 3-channel image by a 3×3 matrix.

    For each pixel ``(r, c)``, computes ``C @ x[r, c, :]``.  Vectorised as
    a single matrix multiplication.

    Parameters
    ----------
    x : np.ndarray
        Image of shape ``(H, W, 3)``.
    C : np.ndarray
        3×3 transformation matrix.

    Returns
    -------
    np.ndarray
        Transformed image of shape ``(H, W, 3)``.
    """
    h, w = x.shape[:2]
    # Reshape to (N, 3), multiply by C^T, reshape back
    pixels = x.reshape(-1, 3)
    result = pixels @ C.T
    return result.reshape(h, w, 3)


def boundary_transform_deblurring(
    img: np.ndarray,
    mode: str,
    boundary_size: int,
    taper: bool = False,
) -> np.ndarray:
    """Add or remove replicate-padded boundary for deblurring.

    Matches MATLAB's ``boundary_transform_deblurring``.

    Parameters
    ----------
    img : np.ndarray
        Input image (2-D or 3-D with channels last).
    mode : str
        ``'add'`` to pad with replicate boundary, ``'rem'`` to crop.
    boundary_size : int
        Number of pixels to pad/crop on each side.
    taper : bool, optional
        If True, apply Gaussian edge tapering after padding. Default False.

    Returns
    -------
    np.ndarray
        Padded or cropped image.
    """
    if mode.lower().startswith("add"):
        mode = "add"
    elif mode.lower().startswith("rem"):
        mode = "rem"
    else:
        mode = "add"

    if mode == "add":
        return _addpad_boundary(img, boundary_size, taper)
    else:
        return _cutpad(img, boundary_size)


def _addpad_boundary(
    f: np.ndarray, p: int, taper: bool
) -> np.ndarray:
    """Add replicate-padded boundary, optionally with Gaussian edge taper.

    MATLAB equivalent::

        f = f([ones(1,p),1:r,r*ones(1,p)], [ones(1,p),1:c,c*ones(1,p)], :);

    Parameters
    ----------
    f : np.ndarray
        Input image.
    p : int
        Padding size on each side.
    taper : bool
        Whether to apply Gaussian edge tapering.

    Returns
    -------
    np.ndarray
        Padded image.
    """
    r, c = f.shape[0], f.shape[1]
    row_idx = np.concatenate([
        np.zeros(p, dtype=int),
        np.arange(r),
        np.full(p, r - 1, dtype=int),
    ])
    col_idx = np.concatenate([
        np.zeros(p, dtype=int),
        np.arange(c),
        np.full(p, c - 1, dtype=int),
    ])
    f = f[np.ix_(row_idx, col_idx)]

    if taper:
        n_channels = f.shape[2] if f.ndim == 3 else 1
        for ch in range(n_channels):
            if f.ndim == 3:
                img_ch = f[:, :, ch]
            else:
                img_ch = f

            # Compute max edgetaper size
            half_min_dim = round(min(img_ch.shape) / 2)
            p_local = p
            if p_local * 2 > half_min_dim:
                p_local = half_min_dim // 2

            # Create Gaussian PSF for tapering: fspecial('gaussian', p*2, p/3)
            size_g = p_local * 2
            sigma_g = p_local / 3.0
            ax = np.arange(size_g) - (size_g - 1) / 2.0
            xx, yy = np.meshgrid(ax, ax)
            gauss_psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_g**2))
            gauss_psf = gauss_psf / gauss_psf.sum()

            img_ch = edgetaper(img_ch, gauss_psf)

            if f.ndim == 3:
                f[:, :, ch] = img_ch
            else:
                f = img_ch

    return f


def _cutpad(f_pad: np.ndarray, p: int) -> np.ndarray:
    """Remove boundary padding.

    MATLAB equivalent::

        f = f_pad(p+(1:r), p+(1:c), :);

    Parameters
    ----------
    f_pad : np.ndarray
        Padded image.
    p : int
        Padding size to remove from each side.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    if f_pad.ndim == 2:
        return f_pad[p:-p, p:-p]
    else:
        return f_pad[p:-p, p:-p, :]
