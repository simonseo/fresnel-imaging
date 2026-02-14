"""Deconvolution algorithms for Fresnel lens imaging.

Modules
-------
utils
    Low-level helpers: ``psf2otf``, ``edgetaper``, ``imconv``, color-space
    transforms, and boundary handling.
solve_image
    Proximal operator for hyper-Laplacian image priors (α = 2/3, 1/2, 1).
pd_joint_deconv
    Primal-dual cross-channel deconvolution (Chambolle–Pock with TV + L1).
fast_deconv
    Single-channel half-quadratic splitting (Krishnan & Fergus, NIPS 2009).
fast_deconv_yuv
    Multi-channel YUV deconvolution (Schuler et al., ICCV 2011).
metrics
    Image quality metrics: SNR, PSNR, SSIM.
"""

from fresnel_imaging.deconvolution.fast_deconv import fast_deconv
from fresnel_imaging.deconvolution.fast_deconv_yuv import fast_deconv_yuv
from fresnel_imaging.deconvolution.metrics import psnr, snr, ssim
from fresnel_imaging.deconvolution.pd_joint_deconv import (
    compute_operator_norm,
    pd_joint_deconv,
)
from fresnel_imaging.deconvolution.solve_image import solve_image
from fresnel_imaging.deconvolution.utils import (
    boundary_transform_deblurring,
    edgetaper,
    imconv,
    img_mult,
    img_to_norm_grayscale,
    psf2otf,
)

__all__ = [
    "boundary_transform_deblurring",
    "compute_operator_norm",
    "edgetaper",
    "fast_deconv",
    "fast_deconv_yuv",
    "imconv",
    "img_mult",
    "img_to_norm_grayscale",
    "pd_joint_deconv",
    "psnr",
    "psf2otf",
    "snr",
    "solve_image",
    "ssim",
]
