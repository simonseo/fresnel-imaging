"""Primal-dual cross-channel deconvolution (Chambolle-Pock).

Implements the joint multi-channel deconvolution algorithm from:
F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
*High-Quality Computational Imaging Through Simple Lenses.* ACM ToG 2013.

Ported from MATLAB code by Felix Heide (fheide@cs.ubc.ca).
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import scipy.sparse.linalg
from numpy.fft import fft2, ifft2

from .utils import edgetaper, imconv, psf2otf

# Channel dict type: {'Image': np.ndarray (2-D), 'K': np.ndarray (PSF)}
ChannelDict = dict[str, np.ndarray]

# Derivative filters (module-level constants)
_DXF = np.array([[-1, 1]])             # 1×2
_DYF = np.array([[-1], [1]])           # 2×1
_DXXF = np.array([[-1, 2, -1]])        # 1×3
_DYYF = np.array([[-1], [2], [-1]])    # 3×1
_DXYF = np.array([[-1, 1], [1, -1]])   # 2×2

_EPS = np.finfo(float).eps


def pd_joint_deconv(
    channels: list[ChannelDict],
    channels_0: list[ChannelDict] | None,
    w_base: np.ndarray,
    max_it: int,
    tol: float,
    verbose: Literal["all", "brief", "none"] = "none",
) -> list[ChannelDict]:
    """Joint multi-channel primal-dual deconvolution.

    Parameters
    ----------
    channels : list[ChannelDict]
        Observed (blurred) channels.  Each dict has keys ``'Image'``
        (2-D ndarray) and ``'K'`` (PSF ndarray, may be ``None``/empty).
    channels_0 : list[ChannelDict] | None
        Initial iterates.  If ``None``, ``channels`` is used.
    w_base : np.ndarray
        Weight matrix.  Each row is
        ``[ch, lambda_res, lambda_tv, lambda_black,
          *lambda_cross_ch, n_detail_layers]``.
    max_it : int
        Maximum Chambolle-Pock iterations per stage.
    tol : float
        Relative convergence tolerance.
    verbose : {'all', 'brief', 'none'}
        Verbosity level.

    Returns
    -------
    list[ChannelDict]
        Deconvolved channels (same structure as input).
    """
    if len(channels) < 1:
        raise ValueError("No valid channels found for deconvolution.")

    if channels_0 is not None and len(channels_0) != len(channels):
        raise ValueError("Initial channels do not match channels.")

    # Normalise verbose
    if isinstance(verbose, str):
        v = verbose.lower()
        if v.startswith("all"):
            verbose = "all"
        elif v.startswith("bri"):
            verbose = "brief"
        else:
            verbose = "none"

    # Initialise
    if channels_0 is None:
        channels_0 = copy.deepcopy(channels)

    db_chs = copy.deepcopy(channels)

    for ch_idx in range(len(channels)):
        k = db_chs[ch_idx].get("K")
        if k is not None and k.size > 0:
            db_chs[ch_idx]["Image"] = channels_0[ch_idx]["Image"].copy()

    # Iterate over rows of w_base
    for s in range(w_base.shape[0]):
        if verbose in ("brief", "all"):
            print(f"\n### Startup iteration {s + 1} ###")

        # MATLAB 1-based channel index → 0-based
        ch_opt = int(w_base[s, 0]) - 1
        w_res_curr = w_base[s, 1]
        w_tv_curr = w_base[s, 2]
        w_black_curr = w_base[s, 3]
        w_cross_curr = w_base[s, 4:-1]
        res_iter = int(w_base[s, -1])

        if res_iter > 0 and np.any(w_cross_curr != 0):
            raise ValueError(
                "Residual iteration with cross channel terms "
                "is not supported"
            )

        # Edge-taper to handle circular boundary conditions
        ks = channels[ch_opt]["K"].shape[0]

        # Deep-copy channels for this stage so padding is local
        ch_work = copy.deepcopy(channels)
        db_work = copy.deepcopy(db_chs)

        for ci in range(len(channels)):
            ch_work[ci]["Image"] = np.pad(
                ch_work[ci]["Image"],
                [(ks, ks), (ks, ks)],
                mode="edge",
            )
            db_work[ci]["Image"] = np.pad(
                db_work[ci]["Image"],
                [(ks, ks), (ks, ks)],
                mode="edge",
            )
            for _ in range(4):
                ch_work[ci]["Image"] = edgetaper(
                    ch_work[ci]["Image"],
                    channels[ch_opt]["K"],
                )
                db_work[ci]["Image"] = edgetaper(
                    db_work[ci]["Image"],
                    channels[ch_opt]["K"],
                )

            ch_work[ci]["Image"] = ch_work[ci]["Image"] + 1.0
            db_work[ci]["Image"] = db_work[ci]["Image"] + 1.0

        # Residual PD deconvolution
        db_work[ch_opt]["Image"] = _residual_pd_deconv(
            ch_work,
            db_work,
            ch_opt,
            w_res_curr,
            w_tv_curr,
            w_black_curr,
            w_cross_curr,
            res_iter,
            tol,
            max_it,
            verbose,
        )

        # Remove padding and offset
        for ci in range(len(channels)):
            ch_work[ci]["Image"] = (
                ch_work[ci]["Image"][ks:-ks, ks:-ks] - 1.0
            )
            db_work[ci]["Image"] = (
                db_work[ci]["Image"][ks:-ks, ks:-ks] - 1.0
            )

        # Write back
        channels = ch_work
        db_chs = db_work

    return db_chs


# ------------------------------------------------------------------ #
#  Residual deconvolution                                             #
# ------------------------------------------------------------------ #

def _residual_pd_deconv(
    channels: list[ChannelDict],
    db_chs: list[ChannelDict],
    ch: int,
    w_res_curr: float,
    w_tv_curr: float,
    w_black_curr: float,
    w_cross_curr: np.ndarray,
    res_iter: int,
    tol: float,
    max_it: int,
    verbose: str,
) -> np.ndarray:
    """Residual deconvolution with detail layers."""
    detail_tol = tol

    for d in range(res_iter + 1):
        if d == 0:
            channels_res = copy.deepcopy(channels)
            x_0 = db_chs[ch]["Image"].copy()
            tol_offset = np.zeros_like(db_chs[ch]["Image"])
        else:
            channels_res = copy.deepcopy(channels)
            residual = channels[ch]["Image"] - imconv(
                db_chs[ch]["Image"], db_chs[ch]["K"], "same"
            )
            channels_res[ch]["Image"] = 1.0 + residual
            x_0 = channels_res[ch]["Image"].copy()
            w_res_curr = w_res_curr * 3.0
            tol_offset = db_chs[ch]["Image"] - 1.0

        x = _pd_channel_deconv(
            channels_res,
            ch,
            x_0,
            db_chs,
            w_res_curr,
            w_cross_curr,
            w_tv_curr,
            w_black_curr,
            max_it,
            detail_tol,
            tol_offset,
            verbose,
        )

        # Threshold: x(x < 1) = 1
        x = np.maximum(x, 1.0)

        if d == 0:
            db_chs[ch]["Image"] = x
        else:
            db_chs[ch]["Image"] = db_chs[ch]["Image"] + (x - 1.0)
            db_chs[ch]["Image"] = np.maximum(db_chs[ch]["Image"], 0.0)

    return db_chs[ch]["Image"]


# ------------------------------------------------------------------ #
#  Single-channel primal-dual deconvolution                           #
# ------------------------------------------------------------------ #

def _pd_channel_deconv(
    channels: list[ChannelDict],
    ch: int,
    x_0: np.ndarray | None,
    db_chs: list[ChannelDict],
    lambda_residual: float,
    lambda_cross_ch: np.ndarray,
    lambda_tv: float,
    lambda_black: float,
    max_it: int,
    tol: float,
    tol_offset: np.ndarray,
    verbose: str,
) -> np.ndarray:
    """Single-channel primal-dual (Chambolle-Pock) minimisation.

    Solves a TV + cross-channel regularised deblurring problem using
    the Chambolle-Pock primal-dual algorithm.
    """
    sizey = channels[ch]["Image"].shape
    otfk = psf2otf(channels[ch]["K"], sizey)
    nomin1 = np.conj(otfk) * fft2(channels[ch]["Image"])
    denom1 = np.abs(otfk) ** 2

    # Operator norm for step-size selection
    op_norm = compute_operator_norm(
        lambda x: _kmult(x, ch, db_chs, lambda_cross_ch, lambda_tv),
        lambda x: _ksmult(x, ch, db_chs, lambda_cross_ch, lambda_tv),
        sizey,
    )

    sigma = 1.0
    tau = 0.7 / (sigma * op_norm**2)
    theta = 1.0

    if x_0 is None:
        x_0 = channels[ch]["Image"].copy()

    f = x_0.copy()
    g = _kmult(f, ch, db_chs, lambda_cross_ch, lambda_tv)
    f1 = f.copy()

    for i in range(1, max_it + 1):
        fold = f.copy()

        # Dual step: g = ProxFS(g + sigma * K(f1))
        g_update = g + sigma * _kmult(
            f1, ch, db_chs, lambda_cross_ch, lambda_tv
        )
        # ProxFS for L1: u / max(1, |u|)
        amp = np.sqrt(g_update**2)
        g = g_update / np.maximum(1.0, amp)

        # Primal step: f = ProxG(f - tau * KS(g))
        ks_g = _ksmult(g, ch, db_chs, lambda_cross_ch, lambda_tv)
        f = _solve_fft(nomin1, denom1, tau, lambda_residual, f - tau * ks_g)

        # Over-relaxation
        f1 = f + theta * (f - fold)

        # Convergence check
        diff = (f + tol_offset) - (fold + tol_offset)
        f_comp = f + tol_offset
        rel_change = (
            np.linalg.norm(diff.ravel())
            / (np.linalg.norm(f_comp.ravel()) + _EPS)
        )

        if verbose in ("brief", "all"):
            print(f"Ch: {ch}, iter {i}, diff {rel_change:.5g}")

        if rel_change < tol:
            break

    return f1


# ------------------------------------------------------------------ #
#  FFT-based data solve                                               #
# ------------------------------------------------------------------ #

def _solve_fft(
    nomin1: np.ndarray,
    denom1: np.ndarray,
    tau: float,
    lam: float,
    f: np.ndarray,
) -> np.ndarray:
    r"""Fast FFT solve for the data-fidelity proximal operator.

    Solves :math:`Ax = b` where
    :math:`A = (\tau \lambda K^H K + I)` and
    :math:`b = \tau \lambda K^H y + f`.

    Uses pre-computed ``nomin1 = conj(OTF) * FFT(y)`` and
    ``denom1 = |OTF|^2``.
    """
    x = (tau * 2 * lam * nomin1 + fft2(f)) / (
        tau * 2 * lam * denom1 + 1.0
    )
    return np.real(ifft2(x))


# ------------------------------------------------------------------ #
#  Forward operator  K                                                #
# ------------------------------------------------------------------ #

def _kmult(
    f: np.ndarray,
    ch: int,
    db_chs: list[ChannelDict],
    lambda_cross_ch: np.ndarray,
    lambda_tv: float,
) -> np.ndarray:
    r"""Forward operator: derivative filters + cross-channel coupling.

    Applies first- and second-order derivative filters to *f*, then
    appends cross-channel gradient coupling terms.  Results are stacked
    along axis 2.

    The derivative filters are flipped (``K[::-1, ::-1]``) before
    convolution in ``'full'`` mode, matching MATLAB's
    ``imconv(f, fliplr(flipud(K)), 'full')``.
    """
    slices: list[np.ndarray] = []

    if lambda_tv > _EPS:
        # dx: fliplr(flipud([-1 1])) = [1 -1]
        fx = imconv(f, _DXF[::-1, ::-1], "full")
        fx = (lambda_tv * 0.5) * fx[:, 1:]  # MATLAB: fx(:, 2:end)

        # dy: fliplr(flipud([-1;1])) = [1;-1]
        fy = imconv(f, _DYF[::-1, ::-1], "full")
        fy = (lambda_tv * 0.5) * fy[1:, :]  # MATLAB: fy(2:end, :)

        sd_w = 0.15
        # dxx
        fxx = imconv(f, _DXXF[::-1, ::-1], "full")
        fxx = (lambda_tv * sd_w) * fxx[:, 2:]  # MATLAB: fxx(:, 3:end)

        # dyy
        fyy = imconv(f, _DYYF[::-1, ::-1], "full")
        fyy = (lambda_tv * sd_w) * fyy[2:, :]  # MATLAB: fyy(3:end, :)

        # dxy
        fxy = imconv(f, _DXYF[::-1, ::-1], "full")
        fxy = (lambda_tv * sd_w) * fxy[1:, 1:]  # MATLAB: fxy(2:end, 2:end)

        slices.extend([fx, fy, fxx, fyy, fxy])

    # Cross-channel terms
    if np.sum(lambda_cross_ch) > _EPS:
        for adj_ch in range(len(db_chs)):
            if adj_ch == ch:
                continue
            k = db_chs[adj_ch].get("K")
            if k is None or k.size == 0:
                continue

            adj_img = db_chs[adj_ch]["Image"]
            lam_c = lambda_cross_ch[adj_ch]

            # Sx: cross-channel x-gradient coupling
            diag_x = imconv(adj_img, _DXF[::-1, ::-1], "full")
            diag_x = diag_x[:, 1:] * f
            conv_x = imconv(f, _DXF[::-1, ::-1], "full")
            sxf = (lam_c * 0.5) * (adj_img * conv_x[:, 1:] - diag_x)

            # Sy: cross-channel y-gradient coupling
            diag_y = imconv(adj_img, _DYF[::-1, ::-1], "full")
            diag_y = diag_y[1:, :] * f
            conv_y = imconv(f, _DYF[::-1, ::-1], "full")
            syf = (lam_c * 0.5) * (adj_img * conv_y[1:, :] - diag_y)

            slices.extend([sxf, syf])

    if not slices:
        return np.zeros((*f.shape, 0))

    return np.stack(slices, axis=2)


# ------------------------------------------------------------------ #
#  Adjoint operator  K*                                               #
# ------------------------------------------------------------------ #

def _ksmult(
    f: np.ndarray,
    ch: int,
    db_chs: list[ChannelDict],
    lambda_cross_ch: np.ndarray,
    lambda_tv: float,
) -> np.ndarray:
    r"""Adjoint of :func:`_kmult`.

    Takes a 3-D stack (output of ``_kmult``) and produces a 2-D image
    by applying the adjoint derivative filters and summing.

    In the adjoint, the filters are applied *without* flipping (original
    orientation), and the output is *truncated* instead of being
    expanded.
    """
    result = np.zeros((f.shape[0], f.shape[1]))

    # Slice index into the 3rd dimension of f
    idx = 0

    if lambda_tv > _EPS:
        # dx adjoint
        fx = imconv((lambda_tv * 0.5) * f[:, :, idx], _DXF, "full")
        fx = fx[:, :-1]  # MATLAB: fx(:, 1:end-1)
        idx += 1

        # dy adjoint
        fy = imconv((lambda_tv * 0.5) * f[:, :, idx], _DYF, "full")
        fy = fy[:-1, :]  # MATLAB: fy(1:end-1, :)
        idx += 1

        sd_w = 0.15
        # dxx adjoint
        fxx = imconv((lambda_tv * sd_w) * f[:, :, idx], _DXXF, "full")
        fxx = fxx[:, :-2]  # MATLAB: fxx(:, 1:end-2)
        idx += 1

        # dyy adjoint
        fyy = imconv((lambda_tv * sd_w) * f[:, :, idx], _DYYF, "full")
        fyy = fyy[:-2, :]  # MATLAB: fyy(1:end-2, :)
        idx += 1

        # dxy adjoint
        fxy = imconv((lambda_tv * sd_w) * f[:, :, idx], _DXYF, "full")
        fxy = fxy[:-1, :-1]  # MATLAB: fxy(1:end-1, 1:end-1)
        idx += 1

        result = fx + fy + fxx + fyy + fxy

    # Cross-channel adjoint terms
    if np.sum(lambda_cross_ch) > _EPS:
        # Make a mutable copy of f for in-place scaling
        f = f.copy()

        for adj_ch in range(len(db_chs)):
            if adj_ch == ch:
                continue
            k = db_chs[adj_ch].get("K")
            if k is None or k.size == 0:
                continue

            adj_img = db_chs[adj_ch]["Image"]
            lam_c = lambda_cross_ch[adj_ch]

            # x-direction adjoint
            f[:, :, idx] = (lam_c * 0.5) * f[:, :, idx]
            fi_x = f[:, :, idx]

            diag_x = imconv(adj_img, _DXF[::-1, ::-1], "full")
            diag_x = diag_x[:, 1:] * fi_x
            conv_x = imconv(adj_img * fi_x, _DXF, "full")
            sxtf = conv_x[:, :-1] - diag_x
            idx += 1

            # y-direction adjoint
            f[:, :, idx] = (lam_c * 0.5) * f[:, :, idx]
            fi_y = f[:, :, idx]

            diag_y = imconv(adj_img, _DYF[::-1, ::-1], "full")
            diag_y = diag_y[1:, :] * fi_y
            conv_y = imconv(adj_img * fi_y, _DYF, "full")
            sytf = conv_y[:-1, :] - diag_y
            idx += 1

            result = result + sxtf + sytf

    return result


# ------------------------------------------------------------------ #
#  Operator norm                                                      #
# ------------------------------------------------------------------ #

def compute_operator_norm(
    a_op: callable,
    as_op: callable,
    sx: tuple[int, ...],
) -> float:
    r"""Compute the operator norm :math:`\|A\| = \sqrt{\lambda_{\max}(A^* A)}`.

    Uses ARPACK (``scipy.sparse.linalg.eigsh``) to find the largest
    eigenvalue of :math:`A^* A` applied as a ``LinearOperator``.

    Parameters
    ----------
    a_op : callable
        Forward operator ``A(x)`` mapping 2-D array → 3-D stack.
    as_op : callable
        Adjoint operator ``A*(x)`` mapping 3-D stack → 2-D array.
    sx : tuple[int, ...]
        Image shape ``(rows, cols)``.

    Returns
    -------
    float
        Operator norm.
    """
    m, n = sx[0], sx[1]
    vec_size = m * n

    def asa_matvec(x: np.ndarray) -> np.ndarray:
        x_img = x.reshape(m, n)
        return as_op(a_op(x_img)).ravel()

    lin_op = scipy.sparse.linalg.LinearOperator(
        shape=(vec_size, vec_size),
        matvec=asa_matvec,
    )

    eigenvalues = scipy.sparse.linalg.eigsh(
        lin_op, k=1, which="LM", tol=1e-3, maxiter=10, return_eigenvectors=False
    )
    return float(np.sqrt(np.abs(eigenvalues[0])))
