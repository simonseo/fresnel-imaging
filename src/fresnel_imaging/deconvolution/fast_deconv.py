"""Single-channel hyper-Laplacian deconvolution.

Port of fast_deconv.m from:
D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
Priors", Proceedings of NIPS 2009.

Uses half-quadratic splitting with continuation to solve:
    min_x  (lambda/2)||Kx - y||^2  +  sum |D_i x|^alpha
where D_i are gradient operators and alpha is the hyper-Laplacian exponent.

Related to the work and code of Wang et al.:
Y. Wang, J. Yang, W. Yin and Y. Zhang, "A New Alternating Minimization
Algorithm for Total Variation Image Reconstruction", SIAM J. Imaging Sci., 2008.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .solve_image import solve_image
from .utils import psf2otf


def compute_denominator(
    y: NDArray[np.floating], k: NDArray[np.floating]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute denominator and part of the numerator for the x-subproblem.

    Implements Equation (3) of the Krishnan & Fergus NIPS 2009 paper.

    Parameters
    ----------
    y : ndarray, shape (M, N)
        Blurry and noisy input image.
    k : ndarray
        Convolution kernel (PSF).

    Returns
    -------
    Nomin1 : ndarray, complex
        ``conj(F(K)) * F(y)`` — numerator contribution from the likelihood.
    Denom1 : ndarray, float
        ``|F(K)|^2`` — denominator contribution from the kernel.
    Denom2 : ndarray, float
        ``|F(D_x)|^2 + |F(D_y)|^2`` — denominator contribution from gradients.
    """
    size_y = (y.shape[0], y.shape[1])
    otfk = psf2otf(k, size_y)
    Nomin1 = np.conj(otfk) * np.fft.fft2(y)
    Denom1 = np.abs(otfk) ** 2

    # Gradient filters in frequency domain
    Denom2 = (
        np.abs(psf2otf(np.array([[1, -1]]), size_y)) ** 2
        + np.abs(psf2otf(np.array([[1], [-1]]), size_y)) ** 2
    )
    return Nomin1, Denom1, Denom2


def fast_deconv(
    yin: NDArray[np.floating],
    k: NDArray[np.floating],
    lambda_param: float,
    alpha: float,
    yout0: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Deconvolve a single-channel image using a hyper-Laplacian prior.

    Solves the optimization problem via half-quadratic splitting with
    continuation on the penalty parameter beta.

    Parameters
    ----------
    yin : ndarray, shape (M, N)
        Observed blurry and noisy grayscale image.
    k : ndarray
        Convolution kernel (must be odd-sized in both dimensions).
    lambda_param : float
        Balances likelihood vs. prior term weighting.
    alpha : float
        Hyper-Laplacian exponent, typically 2/3. Must be in (0, 2].
    yout0 : ndarray, optional
        Initialization for the output. If *None*, ``yin`` is used.

    Returns
    -------
    yout : ndarray, shape (M, N)
        Deconvolved image.

    Raises
    ------
    ValueError
        If the kernel dimensions are not odd.
    """
    # Validate kernel size
    if k.shape[0] % 2 != 1 or k.shape[1] % 2 != 1:
        raise ValueError("Blur kernel k must be odd-sized.")

    # Continuation parameters
    beta: float = 1.0
    beta_rate: float = 2.0 * np.sqrt(2.0)
    beta_max: float = 2.0**8

    # Number of inner iterations per outer iteration
    mit_inn = 1

    m, n = yin.shape[:2]

    # Initialize output
    yout = yout0.copy() if yout0 is not None else yin.copy()

    # Compute constant quantities (Eqn. 3)
    Nomin1, Denom1, Denom2 = compute_denominator(yin, k)

    # x and y gradients with circular boundary conditions
    # MATLAB: diff(yout,1,2) with wrap-around column
    wrap_col = (yout[:, 0] - yout[:, -1])[:, np.newaxis]
    youtx = np.concatenate([np.diff(yout, axis=1), wrap_col], axis=1)
    # MATLAB: diff(yout,1,1) with wrap-around row
    wrap_row = (yout[0, :] - yout[-1, :])[np.newaxis, :]
    youty = np.concatenate([np.diff(yout, axis=0), wrap_row], axis=0)

    # Main continuation loop
    outer_iter = 0
    while beta < beta_max:
        outer_iter += 1

        gamma = beta / lambda_param
        Denom = Denom1 + gamma * Denom2

        for _ in range(mit_inn):
            # ---- w-subproblem: Eqn (5) ----
            Wx = solve_image(youtx, beta, alpha)
            Wy = solve_image(youty, beta, alpha)

            # ---- x-subproblem: Eqn (3) ----
            # Transpose of gradient operators (adjoint with circular boundary)
            # MATLAB: Wxx = [Wx(:,n) - Wx(:,1), -diff(Wx,1,2)]
            Wxx = np.concatenate(
                [(Wx[:, -1] - Wx[:, 0])[:, np.newaxis], -np.diff(Wx, axis=1)], axis=1
            )
            # MATLAB: Wxx = Wxx + [Wy(m,:) - Wy(1,:); -diff(Wy,1,1)]
            Wxx = Wxx + np.concatenate(
                [(Wy[-1, :] - Wy[0, :])[np.newaxis, :], -np.diff(Wy, axis=0)], axis=0
            )

            Fyout = (Nomin1 + gamma * np.fft.fft2(Wxx)) / Denom
            yout = np.real(np.fft.ifft2(Fyout))

            # Update gradients with the new solution
            youtx = np.concatenate(
                [np.diff(yout, axis=1), (yout[:, 0] - yout[:, -1])[:, np.newaxis]], axis=1
            )
            youty = np.concatenate(
                [np.diff(yout, axis=0), (yout[0, :] - yout[-1, :])[np.newaxis, :]], axis=0
            )

        beta *= beta_rate

    return yout
