"""YUV hyper-Laplacian deconvolution with per-channel blur kernels.

Port of fast_deconv_yuv.m — reimplementation of:
C. J. Schuler, M. Hirsch, S. Harmeling and B. Scholkopf:
"Non-stationary Correction of Optical Aberrations", Proceedings of ICCV 2011.

Based on the single-channel code from:
D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
Priors", Proceedings of NIPS 2009.

Extends single-channel deblurring to multi-channel by working in YUV color
space with per-channel blur kernels. The CG solver handles the coupled
system arising from the color-space transformation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .solve_image import solve_image
from .utils import edgetaper, imconv, img_mult

# RGB ↔ YUV conversion matrix (ITU-R BT.601)
RGB_TO_YUV: NDArray[np.float64] = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001],
    ]
)


def _apply_operator(
    x: NDArray[np.floating],
    dxf: NDArray[np.floating],
    dyf: NDArray[np.floating],
    k: list[NDArray[np.floating]],
    C: NDArray[np.floating],
    w_rgb: NDArray[np.floating],
    theta: float,
    lambda_over_beta: float,
    theta_over_beta: float,
) -> NDArray[np.floating]:
    """Compute A*x for the CG linear system.

    A = C' D_x' D_x C  +  C' D_y' D_y C  +  (lambda/beta) K'K  +  (theta/beta)*2*diag(w_rgb)

    Parameters
    ----------
    x : ndarray, shape (M, N, C)
        Current iterate in RGB space.
    dxf, dyf : ndarray
        Horizontal and vertical gradient filters.
    k : list of ndarray
        Per-channel PSFs.
    C : ndarray, shape (3, 3)
        RGB-to-YUV conversion matrix.
    w_rgb : ndarray, shape (C,)
        Per-channel Tikhonov weights.
    theta : float
        Tikhonov regularisation parameter.
    lambda_over_beta : float
        Ratio ``lambda_param / beta``.
    theta_over_beta : float
        Ratio ``theta / beta``.

    Returns
    -------
    Ax : ndarray, shape (M, N, C)
    """
    _m, _n, c = x.shape

    # K'K x  (per-channel convolution)
    x_ktk = np.zeros_like(x)
    for ch in range(c):
        k_flip = np.flip(k[ch])
        x_ktk[:, :, ch] = imconv(imconv(x[:, :, ch], k[ch], "same"), k_flip, "same")

    # Transform to YUV
    x_yuv = img_mult(x, C)

    x_dxtdx = np.zeros_like(x)
    x_dytdy = np.zeros_like(x)
    dxf_flip = np.flip(dxf)
    dyf_flip = np.flip(dyf)

    for ch in range(c):
        # D_x' D_x  in YUV
        tmp = imconv(x_yuv[:, :, ch], dxf, "full")
        tmp = imconv(tmp[:, :-1], dxf_flip, "full")
        x_dxtdx[:, :, ch] = tmp[:, 1:]

        # D_y' D_y  in YUV
        tmp = imconv(x_yuv[:, :, ch], dyf, "full")
        tmp = imconv(tmp[:-1, :], dyf_flip, "full")
        x_dytdy[:, :, ch] = tmp[1:, :]

    # Back to RGB
    x_dxtdx = img_mult(x_dxtdx, C.T)
    x_dytdy = img_mult(x_dytdy, C.T)

    # Tikhonov term
    rb_x = np.zeros_like(x)
    for ch in range(c):
        rb_x[:, :, ch] = theta_over_beta * 2.0 * w_rgb[ch] * x[:, :, ch]

    return x_dxtdx + x_dytdy + lambda_over_beta * x_ktk + rb_x


def solve_cg_subproblem(
    lambda_param: float,
    beta: float,
    dxf: NDArray[np.floating],
    dyf: NDArray[np.floating],
    k: list[NDArray[np.floating]],
    b: NDArray[np.floating],
    C: NDArray[np.floating],
    w_rgb: NDArray[np.floating],
    theta: float,
    x_0: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Conjugate-gradient solver for the quadratic x-subproblem.

    Solves ``A x = b`` where:
        A = C' D_x' D_x C  +  C' D_y' D_y C  +  (lambda/beta) K'K
            + (theta/beta)*2*diag(w_rgb)

    Parameters
    ----------
    lambda_param : float
        Likelihood weighting.
    beta : float
        Current continuation parameter (outer loop).
    dxf, dyf : ndarray
        Gradient filter kernels ``[1, -1]`` and ``[[1], [-1]]``.
    k : list of ndarray
        Per-channel PSFs.
    b : ndarray, shape (M, N, C)
        Right-hand side of the linear system.
    C : ndarray, shape (3, 3)
        RGB-to-YUV conversion matrix.
    w_rgb : ndarray, shape (C,)
        Per-channel Tikhonov weights.
    theta : float
        Tikhonov regularisation parameter.
    x_0 : ndarray, shape (M, N, C)
        Initial guess.

    Returns
    -------
    x : ndarray, shape (M, N, C)
        Approximate solution.
    """
    cg_tol = 1e-5
    cg_iter = min(50, x_0.size)

    lambda_over_beta = lambda_param / beta
    theta_over_beta = theta / beta

    x = x_0.copy()

    # Initial residual: r = b - A*x
    Ax = _apply_operator(x, dxf, dyf, k, C, w_rgb, theta, lambda_over_beta, theta_over_beta)
    r = b - Ax

    p: NDArray[np.floating] | None = None
    rho_prev: float = 0.0

    for iteration in range(cg_iter):
        rho = float(np.dot(r.ravel(), r.ravel()))

        if iteration > 0:
            # CG direction coefficient — renamed from MATLAB's `beta` to avoid
            # shadowing the outer continuation parameter.
            cg_beta = rho / rho_prev
            p = r + cg_beta * p  # type: ignore[operator]
        else:
            p = r.copy()

        # Compute A*p
        Ap = _apply_operator(p, dxf, dyf, k, C, w_rgb, theta, lambda_over_beta, theta_over_beta)

        # CG step — MATLAB reuses `alpha` variable name for this; we use
        # `cg_alpha` to keep the outer hyper-Laplacian exponent untouched.
        cg_alpha = rho / float(np.dot(p.ravel(), Ap.ravel()))
        x = x + cg_alpha * p
        r = r - cg_alpha * Ap

        rho_prev = rho

        # Convergence check
        if np.linalg.norm(r.ravel()) <= cg_tol:
            break

    return x


def fast_deconv_yuv(
    yin: NDArray[np.floating],
    k: list[NDArray[np.floating]],
    lambda_param: float,
    rho_yuv: NDArray[np.floating] | list[float],
    w_rgb: NDArray[np.floating] | list[float],
    theta: float,
    alpha: float,
    yout0: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Deconvolve a multi-channel image using a YUV hyper-Laplacian prior.

    Parameters
    ----------
    yin : ndarray, shape (M, N, C)
        Observed blurry and noisy RGB image.
    k : list of ndarray
        Per-channel blur kernels (list of 2-D arrays, one per channel).
        Each kernel must be odd-sized.
    lambda_param : float
        Likelihood weighting parameter.
    rho_yuv : array-like, shape (C,)
        Per-channel (YUV) hyper-Laplacian prior weights.
    w_rgb : array-like, shape (C,)
        Per-channel Tikhonov weights in RGB space.
    theta : float
        Tikhonov regularisation strength.
    alpha : float
        Hyper-Laplacian exponent (typically 0.65).
    yout0 : ndarray, optional
        Initialisation for the deblurred image. Defaults to ``yin``.

    Returns
    -------
    yout : ndarray, shape (M', N', C)
        Deconvolved image (padding stripped).

    Raises
    ------
    ValueError
        If any kernel has even dimensions.
    """
    rho_yuv_arr = np.asarray(rho_yuv, dtype=np.float64)
    w_rgb_arr = np.asarray(w_rgb, dtype=np.float64)
    C = RGB_TO_YUV

    # ------------------------------------------------------------------ #
    # Prepare y: pad + edgetaper                                          #
    # ------------------------------------------------------------------ #
    # Compute max blur radius across all channels
    ks = max(max(ki.shape[0], ki.shape[1]) for ki in k)
    ks = max(0, ks // 2)

    c = yin.shape[2] if yin.ndim == 3 else 1

    # Pad with replicate boundary
    yin_padded = np.pad(
        yin,
        ((ks, ks), (ks, ks), (0, 0)) if yin.ndim == 3 else ((ks, ks), (ks, ks)),
        mode="edge",
    )

    # Edgetaper (4 passes, matching MATLAB)
    for _pass in range(4):
        for ch in range(c):
            yin_padded[:, :, ch] = edgetaper(yin_padded[:, :, ch], k[ch])

    yin_work = yin_padded

    # ------------------------------------------------------------------ #
    # Actual minimisation                                                 #
    # ------------------------------------------------------------------ #
    # Continuation parameters
    beta: float = 1.0
    beta_rate: float = 2.0 * np.sqrt(2.0)
    beta_max: float = 2.0**8
    mit_inn = 1

    m, n = yin_work.shape[:2]
    c = yin_work.shape[2] if yin_work.ndim == 3 else 1

    # Validate kernel sizes
    for ch, ki in enumerate(k):
        if ki.shape[0] % 2 != 1 or ki.shape[1] % 2 != 1:
            raise ValueError(f"Blur kernel for channel {ch} must be odd-sized.")

    # Initialise output
    yout = yout0.copy() if yout0 is not None else yin_work.copy()

    # K' * y  (pre-computed, constant)
    KtB = np.zeros_like(yin_work)
    for ch in range(c):
        k_flip = np.flip(k[ch])
        KtB[:, :, ch] = imconv(yin_work[:, :, ch], k_flip, "same")

    # Gradient filters
    dxf = np.array([[1, -1]], dtype=np.float64)
    dyf = np.array([[1], [-1]], dtype=np.float64)

    # Pre-allocate work arrays
    youtx = np.zeros_like(yin_work)
    youty = np.zeros_like(yin_work)
    Wx = np.zeros_like(yout)
    Wy = np.zeros_like(yout)
    Wxtx = np.zeros_like(yout)
    Wyty = np.zeros_like(yout)

    dxf_flip = np.flip(dxf)
    dyf_flip = np.flip(dyf)

    # Convert initial yout to YUV and compute gradients
    yout_yuv = img_mult(yout, C)
    for ch in range(c):
        tmp = imconv(yout_yuv[:, :, ch], dxf, "full")
        youtx[:, :, ch] = tmp[:, :-1]

        tmp = imconv(yout_yuv[:, :, ch], dyf, "full")
        youty[:, :, ch] = tmp[:-1, :]

    # ------------------------------------------------------------------ #
    # Main continuation loop                                              #
    # ------------------------------------------------------------------ #
    while beta < beta_max:
        for _ in range(mit_inn):
            # ---- w-subproblem (Eqn 5) in YUV space ---- #
            for ch in range(c):
                Wx[:, :, ch] = solve_image(youtx[:, :, ch], beta / rho_yuv_arr[ch], alpha)
                Wy[:, :, ch] = solve_image(youty[:, :, ch], beta / rho_yuv_arr[ch], alpha)

            # ---- x-subproblem (Eqn 3) ---- #
            # Compute adjoint gradient terms in YUV
            for ch in range(c):
                tmp = imconv(Wx[:, :, ch], dxf_flip, "full")
                Wxtx[:, :, ch] = tmp[:, 1:]

                tmp = imconv(Wy[:, :, ch], dyf_flip, "full")
                Wyty[:, :, ch] = tmp[1:, :]

            # Apply C' (YUV → RGB)
            Wxtx_rgb = img_mult(Wxtx, C.T)
            Wyty_rgb = img_mult(Wyty, C.T)

            Wxx = Wxtx_rgb + Wyty_rgb

            # RHS of linear system
            rhs = Wxx + (lambda_param / beta) * KtB

            # Solve with CG
            yout = solve_cg_subproblem(
                lambda_param, beta, dxf, dyf, k, rhs, C, w_rgb_arr, theta, yout
            )

            # Update gradients
            yout_yuv = img_mult(yout, C)
            for ch in range(c):
                tmp = imconv(yout_yuv[:, :, ch], dxf, "full")
                youtx[:, :, ch] = tmp[:, :-1]

                tmp = imconv(yout_yuv[:, :, ch], dyf, "full")
                youty[:, :, ch] = tmp[:-1, :]

        beta *= beta_rate

    # ------------------------------------------------------------------ #
    # Remove padding                                                      #
    # ------------------------------------------------------------------ #
    yout = yout[ks : m - ks, ks : n - ks, :]

    return yout
