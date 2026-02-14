"""Command-line entry point for Fresnel lens deblurring.

Replicates the workflow of ``deblur_from_blurred.m``:

1. Load image, resize, normalise to [0, 1], apply inverse gamma.
2. Build per-channel blur kernels from a kernel image with increasing sizes.
3. Add Gaussian noise to the blurred image.
4. Run deconvolution (proposed cross-channel, hyper-Laplacian, and/or YUV).
5. Compute PSNR, apply gamma correction, save results.

Usage::

    fresnel-deblur --image images/houses_big.jpg --kernel kernels/fading.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .deconvolution.fast_deconv import fast_deconv
from .deconvolution.fast_deconv_yuv import fast_deconv_yuv
from .deconvolution.metrics import psnr
from .deconvolution.pd_joint_deconv import pd_joint_deconv
from .deconvolution.utils import edgetaper


def _img_to_norm_grayscale(img: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert an image to normalised [0, 1] grayscale (float64).

    Port of ``img_to_norm_grayscale.m``.
    """
    if img.ndim == 3 and img.shape[2] > 1:
        img = np.mean(img, axis=2)  # simple grayscale (matches rgb2gray approx)
    img = img.astype(np.float64)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-12:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def _load_image(path: str | Path) -> NDArray[np.float64]:
    """Load an image as float64 RGB array with shape (H, W, 3)."""
    pil_img = Image.open(path).convert("RGB")
    return np.asarray(pil_img, dtype=np.float64)


def _load_kernel_image(path: str | Path) -> NDArray[np.float64]:
    """Load a kernel image as float64 grayscale array."""
    pil_img = Image.open(path).convert("L")
    return np.asarray(pil_img, dtype=np.float64)


def _build_kernels(
    kernel_img: NDArray[np.float64],
    blur_size: int,
    inc_blur_exp: float,
    n_channels: int,
) -> tuple[list[NDArray[np.float64]], float]:
    """Build per-channel blur kernels with increasing sizes.

    The first channel gets a delta kernel (sharp), channels 2+ get
    progressively larger blurs.

    Parameters
    ----------
    kernel_img : ndarray
        Base kernel image to resize.
    blur_size : int
        Base kernel size (must be odd).
    inc_blur_exp : float
        Exponential growth factor for kernel size per channel.
    n_channels : int
        Number of image channels.

    Returns
    -------
    K_blur : list of ndarray
        Per-channel normalised kernels.
    kernel_var : float
        Variance of noise added to kernels (for theta computation).
    """
    K_blur: list[NDArray[np.float64]] = []

    # Compute max blur size (for display / theta calculation)
    max_blur_size = round(blur_size * inc_blur_exp ** (n_channels - 1))
    max_blur_size = max_blur_size + (1 - max_blur_size % 2)  # ensure odd

    for ch in range(n_channels):
        curr_blur_size = round(blur_size * inc_blur_exp**ch)
        curr_blur_size = curr_blur_size + (1 - curr_blur_size % 2)  # ensure odd

        # Resize kernel image to current size
        k_ch = cv2.resize(
            kernel_img,
            (curr_blur_size, curr_blur_size),
            interpolation=cv2.INTER_CUBIC,
        )
        k_ch = _img_to_norm_grayscale(k_ch)
        k_sum = k_ch.sum()
        if k_sum > 0:
            k_ch = k_ch / k_sum
        K_blur.append(k_ch)

    # First channel = delta kernel (sharp)
    curr_blur_size_ch0 = blur_size + (1 - blur_size % 2)
    curr_blur_radius = curr_blur_size_ch0 // 2
    k0 = np.zeros((curr_blur_size_ch0, curr_blur_size_ch0), dtype=np.float64)
    k0[curr_blur_radius, curr_blur_radius] = 1.0
    K_blur[0] = k0

    kernel_var = 1e-7
    return K_blur, kernel_var


def _add_kernel_noise(
    K_blur: list[NDArray[np.float64]],
    kernel_var: float,
) -> list[NDArray[np.float64]]:
    """Add Gaussian noise to kernels and re-normalise."""
    noisy: list[NDArray[np.float64]] = []
    for ch, k in enumerate(K_blur):
        noise_std = np.sqrt(kernel_var / (ch + 1) ** 2)
        k_noisy = k + np.random.default_rng().normal(0, noise_std, k.shape)
        k_noisy = np.clip(k_noisy, 0, None)
        k_sum = k_noisy.sum()
        if k_sum > 0:
            k_noisy = k_noisy / k_sum
        noisy.append(k_noisy)
    return noisy


def _blur_image(
    I_sharp: NDArray[np.float64],
    K_blur: list[NDArray[np.float64]],
    noise_sd: float = 0.005,
) -> NDArray[np.float64]:
    """Apply per-channel blur and additive Gaussian noise."""
    I_blurred = np.zeros_like(I_sharp)
    rng = np.random.default_rng()
    for ch in range(I_sharp.shape[2]):
        # MATLAB: imfilter(I_sharp(:,:,ch), K_blur{ch}, 'conv', 'symmetric')
        I_blurred[:, :, ch] = cv2.filter2D(
            I_sharp[:, :, ch],
            -1,
            K_blur[ch],
            borderType=cv2.BORDER_REFLECT,
        )
        # Add Gaussian noise
        I_blurred[:, :, ch] += rng.normal(0, noise_sd, I_blurred[:, :, ch].shape)

    return I_blurred


def _run_proposed(
    channel_patches: list[dict[str, NDArray[np.float64]]],
    lambda_startup: NDArray[np.float64],
) -> list[dict[str, NDArray[np.float64]]]:
    """Run the proposed cross-channel deconvolution (pd_joint_deconv)."""
    print("\nComputing cross-channel deconvolution ...\n")
    result = pd_joint_deconv(
        channel_patches,
        [],
        lambda_startup,
        max_iter=200,
        tol=1e-4,
        verbose="brief",
    )
    return result


def _run_hyp(
    channel_patches: list[dict[str, NDArray[np.float64]]],
    lambda_hyp: float = 2000.0,
) -> list[dict[str, NDArray[np.float64]]]:
    """Run per-channel hyper-Laplacian deconvolution."""
    print("\nComputing hyperlaplacian naive deconvolution ...\n")
    result: list[dict[str, NDArray[np.float64]]] = []
    for patch in channel_patches:
        img_et = edgetaper(patch["Image"], patch["K"])
        deconv = fast_deconv(img_et, patch["K"], lambda_hyp, 2.0 / 3.0)
        result.append({"Image": deconv, "K": patch["K"]})
    return result


def _run_yuv(
    channel_patches: list[dict[str, NDArray[np.float64]]],
    max_blur_size: int,
    kernel_var: float,
) -> list[dict[str, NDArray[np.float64]]]:
    """Run YUV hyper-Laplacian deconvolution."""
    print("\nComputing hyperlaplacian YUV deconvolution ...\n")

    n_ch = len(channel_patches)
    h, w = channel_patches[0]["Image"].shape[:2]

    # Assemble 3-D image and kernel list
    y = np.zeros((h, w, n_ch), dtype=np.float64)
    k_list: list[NDArray[np.float64]] = []
    for ch in range(n_ch):
        y[:, :, ch] = channel_patches[ch]["Image"]
        k_list.append(channel_patches[ch]["K"])

    # Parameters from Schuler et al. ICCV 2011
    lambda_yuv = 2e3
    alpha_yuv = 0.65
    rho_yuv = [0.1, 1.0, 1.0]
    w_rgb = [0.25, 0.5, 0.25]
    theta = max_blur_size**2 * kernel_var * 1e3

    x = fast_deconv_yuv(y, k_list, lambda_yuv, rho_yuv, w_rgb, theta, alpha_yuv)

    result: list[dict[str, NDArray[np.float64]]] = []
    for ch in range(n_ch):
        result.append({"Image": x[:, :, ch], "K": k_list[ch]})
    return result


def _compute_psnr(
    I_sharp: NDArray[np.float64],
    I_deconv: NDArray[np.float64],
    psnr_pad: int,
) -> float:
    """Compute PSNR matching the MATLAB implementation."""
    return psnr(I_deconv, I_sharp, crop_pad=psnr_pad)


def _apply_gamma_and_save(
    img: NDArray[np.float64],
    path: Path,
) -> None:
    """Clip negatives, apply gamma correction (^0.5), and save as PNG."""
    img_out = np.clip(img, 0.0, None)
    img_out = img_out ** 0.5
    img_out = np.clip(img_out, 0.0, 1.0)
    # Convert to 16-bit for high-quality output
    img_16 = (img_out * 65535).astype(np.uint16)
    if img_16.ndim == 3:
        pil_img = Image.fromarray(img_16, mode="RGB")
    else:
        pil_img = Image.fromarray(img_16, mode="I;16")
    pil_img.save(str(path))


def _gather_result(
    channel_patches: list[dict[str, NDArray[np.float64]]],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Assemble per-channel results into a single image array."""
    out = np.zeros(shape, dtype=np.float64)
    for ch, patch in enumerate(channel_patches):
        out[:, :, ch] = patch["Image"]
    return out


def main(argv: list[str] | None = None) -> None:
    """Run the deblurring pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Fresnel lens deblurring pipeline (port of deblur_from_blurred.m)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="images/houses_big.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="kernels/fading.png",
        help="Path to the kernel image.",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=15,
        help="Base blur kernel size (must be odd). Default: 15.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "ours", "hyp", "yuv"],
        default="all",
        help="Deconvolution method(s) to run. Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output images. Default: current directory.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load and prepare image
    # ------------------------------------------------------------------
    img = _load_image(args.image)
    # Resize to 15%
    h, w = img.shape[:2]
    new_h, new_w = int(h * 0.15), int(w * 0.15)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.float64)

    # Normalise to [0, 1]
    img = img / img.max() if img.max() > 0 else img

    # Apply inverse gamma
    img = img**2.0

    print(f"Processing {args.image} with size {img.shape[1]} x {img.shape[0]}")
    I_sharp = img.copy()

    # ------------------------------------------------------------------
    # Build kernels
    # ------------------------------------------------------------------
    kernel_img = _load_kernel_image(args.kernel)
    n_channels = img.shape[2]
    inc_blur_exp = 1.7

    K_blur, kernel_var = _build_kernels(
        kernel_img, args.blur_size, inc_blur_exp, n_channels
    )
    K_blur_orig = [k.copy() for k in K_blur]

    # Max blur size for theta computation
    max_blur_size = round(args.blur_size * inc_blur_exp ** (n_channels - 1))
    max_blur_size = max_blur_size + (1 - max_blur_size % 2)

    # ------------------------------------------------------------------
    # Blur + noise
    # ------------------------------------------------------------------
    I_blurred = _blur_image(I_sharp, K_blur, noise_sd=0.005)

    # Add noise to kernels
    K_blur = _add_kernel_noise(K_blur, kernel_var)

    # ------------------------------------------------------------------
    # Prepare channel patches
    # ------------------------------------------------------------------
    channel_patches: list[dict[str, NDArray[np.float64]]] = []
    for ch in range(n_channels):
        channel_patches.append({"Image": I_blurred[:, :, ch], "K": K_blur[ch]})

    # ------------------------------------------------------------------
    # Lambda startup matrix for proposed method
    # ------------------------------------------------------------------
    lambda_startup = np.array(
        [
            [1, 300, 1.0, 0.0, 0.0, 0.0, 0.0, 1],
            [2, 750, 0.5, 0.0, 1.0, 0.0, 0.0, 0],
            [3, 750, 0.5, 0.0, 1.0, 0.0, 0.0, 0],
        ],
        dtype=np.float64,
    )

    # PSNR crop pad
    psnr_pad = round(K_blur_orig[0].shape[0] * 1.5)

    # ------------------------------------------------------------------
    # Run method(s)
    # ------------------------------------------------------------------
    run_ours = args.method in ("all", "ours")
    run_hyp = args.method in ("all", "hyp")
    run_yuv = args.method in ("all", "yuv")

    if run_ours:
        result_ours = _run_proposed(channel_patches, lambda_startup)
        I_deconv_ours = _gather_result(result_ours, I_sharp.shape)
        psnr_ours = _compute_psnr(I_sharp, I_deconv_ours, psnr_pad)
        print(f"Proposed method PSNR: {psnr_ours:.5g} dB")
        _apply_gamma_and_save(I_deconv_ours, out_dir / "I_deconv_ours.png")

    if run_hyp:
        result_hyp = _run_hyp(channel_patches)
        I_deconv_hyp = _gather_result(result_hyp, I_sharp.shape)
        psnr_hyp = _compute_psnr(I_sharp, I_deconv_hyp, psnr_pad)
        print(f"Hyperlaplacian deconv PSNR: {psnr_hyp:.5g} dB")
        _apply_gamma_and_save(I_deconv_hyp, out_dir / "I_deconv_hyp.png")

    if run_yuv:
        result_yuv = _run_yuv(channel_patches, max_blur_size, kernel_var)
        I_deconv_yuv = _gather_result(result_yuv, I_sharp.shape)
        psnr_yuv = _compute_psnr(I_sharp, I_deconv_yuv, psnr_pad)
        print(f"YUV deconv PSNR: {psnr_yuv:.5g} dB")
        _apply_gamma_and_save(I_deconv_yuv, out_dir / "I_deconv_yuv.png")

    # Save reference images
    _apply_gamma_and_save(I_sharp, out_dir / "I_sharp.png")
    _apply_gamma_and_save(I_blurred, out_dir / "I_blurred.png")

    print("Done. Results saved to", out_dir)


if __name__ == "__main__":
    main()
