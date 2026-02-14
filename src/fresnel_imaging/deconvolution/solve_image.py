"""Proximal operator solver for hyper-Laplacian priors.

Solves ``min |w|^alpha + (beta/2) * (w - v)^2`` component-wise using a
lookup-table (LUT) approach.  Exact closed-form solutions are used for
special values of alpha:

* alpha = 1   : soft-thresholding
* alpha = 2/3 : quartic via Ferrari's method
* alpha = 1/2 : cubic via Cardano's formula

For general alpha, Newton-Raphson iteration is used.

Ported from MATLAB code by Dilip Krishnan / Rob Fergus / Felix Heide.
"""

from __future__ import annotations

import numpy as np

# Module-level LUT cache (replaces MATLAB persistent variables).
# Maps ``(beta, alpha)`` → precomputed ``w`` values over a grid ``xx``.
_LUT_RANGE = 10.0
_LUT_STEP = 0.0001
_lut_cache: dict[tuple[float, float], np.ndarray] = {}
_xx: np.ndarray | None = None


def _get_xx() -> np.ndarray:
    """Lazily create the shared LUT sample grid."""
    global _xx  # noqa: PLW0603
    if _xx is None:
        _xx = np.arange(-_LUT_RANGE, _LUT_RANGE + _LUT_STEP / 2, _LUT_STEP)
    return _xx


def solve_image(
    v: np.ndarray, beta: float, alpha: float
) -> np.ndarray:
    r"""Solve the proximal problem component-wise via LUT interpolation.

    .. math::

        \min_w \; |w|^\alpha + \frac{\beta}{2} (w - v)^2

    On the first call for a given ``(beta, alpha)`` pair the LUT is built;
    subsequent calls reuse it.

    Parameters
    ----------
    v : np.ndarray
        Input array (any shape).
    beta : float
        Data-fidelity weight (> 0).
    alpha : float
        Sparsity exponent (e.g. 1, 2/3, 1/2).

    Returns
    -------
    np.ndarray
        Optimal ``w``, same shape as *v*.
    """
    xx = _get_xx()
    key = (beta, alpha)

    if key not in _lut_cache:
        _lut_cache[key] = _compute_w(xx, beta, alpha)

    lut_w = _lut_cache[key]
    w = np.interp(v.ravel(), xx, lut_w)
    return w.reshape(v.shape)


# ------------------------------------------------------------------ #
#  Dispatcher                                                         #
# ------------------------------------------------------------------ #

def _compute_w(
    v: np.ndarray, beta: float, alpha: float
) -> np.ndarray:
    """Dispatch to the specialised solver for the given alpha."""
    if abs(alpha - 1.0) < 1e-9:
        return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < 1e-9:
        return _compute_w23(v, beta)
    if abs(alpha - 0.5) < 1e-9:
        return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)


# ------------------------------------------------------------------ #
#  alpha = 1  (soft threshold)                                        #
# ------------------------------------------------------------------ #

def _compute_w1(v: np.ndarray, beta: float) -> np.ndarray:
    r"""Soft thresholding: :math:`\max(|v| - 1/\beta, 0) \cdot \text{sign}(v)`."""
    return np.maximum(np.abs(v) - 1.0 / beta, 0.0) * np.sign(v)


# ------------------------------------------------------------------ #
#  alpha = 2/3  (quartic via Ferrari's method)                        #
# ------------------------------------------------------------------ #

def _compute_w23(v: np.ndarray, beta: float) -> np.ndarray:
    r"""Quartic solver for :math:`\alpha = 2/3` using Ferrari's method.

    Follows the derivation from
    `Wikipedia: Quartic equation (Ferrari's method)
    <https://en.wikipedia.org/wiki/Quartic_equation>`_ with coefficients
    passed through Mathematica (see ``quartic_solution.nb`` in the original
    MATLAB distribution).
    """
    epsilon = 1e-6

    k = 8.0 / (27.0 * beta**3)
    m = np.full_like(v, k)

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    m2 = m * m
    m3 = m2 * m

    # Ferrari coefficients (reusing names from original MATLAB)
    alpha_f = -1.125 * v2
    beta_f = 0.25 * v3

    q = -0.125 * (m * v2)
    disc_arg = (-m3 / 27.0 + (m2 * v4) / 256.0).astype(complex)
    r1 = -q / 2.0 + np.sqrt(disc_arg)

    u = np.exp(np.log(r1.astype(complex)) / 3.0)
    y = 2.0 * (-5.0 / 18.0 * alpha_f + u + (m / (3.0 * u)))

    w_sqrt = np.sqrt((alpha_f / 3.0 + y).astype(complex))

    # Four roots
    root = np.zeros((*v.shape, 4), dtype=complex)
    inner_p = -(alpha_f + y + beta_f / w_sqrt)
    inner_m = -(alpha_f + y - beta_f / w_sqrt)

    root[..., 0] = 0.75 * v + 0.5 * (w_sqrt + np.sqrt(inner_p))
    root[..., 1] = 0.75 * v + 0.5 * (w_sqrt - np.sqrt(inner_p))
    root[..., 2] = 0.75 * v + 0.5 * (-w_sqrt + np.sqrt(inner_m))
    root[..., 3] = 0.75 * v + 0.5 * (-w_sqrt - np.sqrt(inner_m))

    # Pick the correct root (including zero option).
    # Real roots in (|v|/2, |v|) with the same sign as v.
    v_rep = np.broadcast_to(v[..., np.newaxis], root.shape)
    sv = np.sign(v_rep)
    rsv = np.real(root) * sv

    valid = (
        (np.abs(np.imag(root)) < epsilon)
        & (rsv > np.abs(v_rep) / 2.0)
        & (rsv < np.abs(v_rep))
    )
    # Among valid roots pick the largest (descending sort, take first)
    scores = valid * rsv
    # Sort descending along last axis
    idx = np.argsort(-scores, axis=-1)
    best = np.take_along_axis(scores, idx, axis=-1)[..., 0]
    return best * np.sign(v)


# ------------------------------------------------------------------ #
#  alpha = 1/2  (cubic via Cardano)                                   #
# ------------------------------------------------------------------ #

def _compute_w12(v: np.ndarray, beta: float) -> np.ndarray:
    r"""Cubic solver for :math:`\alpha = 1/2` using Cardano's formula."""
    epsilon = 1e-6

    k = -0.25 / beta**2
    m = np.full_like(v, k) * np.sign(v)

    v2 = v * v
    v3 = v2 * v

    t1 = (2.0 / 3.0) * v

    # Cube-root argument (complex arithmetic)
    disc = 27.0 * m**2 + 4.0 * m * v3
    sqrt_disc = np.sqrt(disc.astype(complex))
    arg = -27.0 * m - 2.0 * v3 + 3.0 * np.sqrt(3.0) * sqrt_disc
    t2 = np.exp(np.log(arg.astype(complex)) / 3.0)

    t3 = v2 / t2

    cbrt2 = 2.0 ** (1.0 / 3.0)
    sqrt3j = 1j * np.sqrt(3.0)

    root = np.zeros((*v.shape, 3), dtype=complex)
    root[..., 0] = t1 + (cbrt2 / 3.0) * t3 + t2 / (3.0 * cbrt2)
    root[..., 1] = (
        t1
        - ((1.0 + sqrt3j) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3
        - ((1.0 - sqrt3j) / (6.0 * cbrt2)) * t2
    )
    root[..., 2] = (
        t1
        - ((1.0 - sqrt3j) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3
        - ((1.0 + sqrt3j) / (6.0 * cbrt2)) * t2
    )

    # Catch NaN/Inf from 0/0 cases
    root = np.where(np.isnan(root) | np.isinf(root), 0.0, root)

    # Pick the correct root: real, in (2|v|/3, |v|), same sign as v.
    v_rep = np.broadcast_to(v[..., np.newaxis], root.shape)
    sv = np.sign(v_rep)
    rsv = np.real(root) * sv

    valid = (
        (np.abs(np.imag(root)) < epsilon)
        & (rsv > 2.0 * np.abs(v_rep) / 3.0)
        & (rsv < np.abs(v_rep))
    )
    scores = valid * rsv
    idx = np.argsort(-scores, axis=-1)
    best = np.take_along_axis(scores, idx, axis=-1)[..., 0]
    return best * np.sign(v)


# ------------------------------------------------------------------ #
#  General alpha (Newton-Raphson)                                     #
# ------------------------------------------------------------------ #

def _newton_w(
    v: np.ndarray, beta: float, alpha: float
) -> np.ndarray:
    r"""Newton-Raphson solver for general :math:`\alpha`.

    Finds roots of
    :math:`\alpha |w|^{\alpha-1} \text{sign}(w) + \beta (w - v) = 0`
    with 4 iterations, then checks whether the zero solution is better.
    """
    x = v.copy()

    for _ in range(4):
        fd = alpha * np.sign(x) * np.abs(x) ** (alpha - 1) + beta * (x - v)
        fdd = alpha * (alpha - 1) * np.abs(x) ** (alpha - 2) + beta
        x = x - fd / fdd

    x = np.where(np.isnan(x), 0.0, x)

    # Check whether zero is a better solution
    cost_zero = (beta / 2.0) * v**2
    cost_x = np.abs(x) ** alpha + (beta / 2.0) * (x - v) ** 2
    return np.where(cost_x < cost_zero, x, 0.0)
