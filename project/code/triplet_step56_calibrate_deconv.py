import json
from pathlib import Path

import cv2
import numpy as np

from fresnel_imaging.calibration.pipeline import (
    apply_spatially_varying_deconv,
    calibrate_fresnel_lens,
    save_calibration,
)
from fresnel_imaging.deconvolution.fast_deconv import fast_deconv

ROOT = Path("/Users/sseo/Documents/fresnel-imaging")
STEP4_DIR = ROOT / "project" / "assets" / "lenses" / "triplet_a" / "step4"
STEP56_DIR = ROOT / "project" / "assets" / "lenses" / "triplet_a" / "step5_step6"
STEP56_DIR.mkdir(parents=True, exist_ok=True)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


blurred_u8 = _load_rgb(STEP4_DIR / "blurred_scene.png")
reference_u8 = _load_rgb(STEP4_DIR / "reference_scene.png")

cal = calibrate_fresnel_lens(
    blurred_image=blurred_u8,
    reference_image=reference_u8,
    psf_size=31,
    patch_size=192,
    method="wiener",
    per_channel=False,
    undistort=False,
)

save_calibration(
    filepath=STEP56_DIR / "calibration_triplet_a",
    camera_matrix=cal["camera_matrix"],
    dist_coeffs=cal["dist_coeffs"],
    psf_grid=cal["psf_grid"],
    positions=cal["positions"],
)

blurred_gray = cv2.cvtColor(blurred_u8, cv2.COLOR_RGB2GRAY)
reference_gray = cv2.cvtColor(reference_u8, cv2.COLOR_RGB2GRAY)
blurred_gray = blurred_gray.astype(np.float64) / 255.0
reference_gray = reference_gray.astype(np.float64) / 255.0


def _deconv_fn(patch: np.ndarray, psf: np.ndarray) -> np.ndarray:
    return fast_deconv(patch, psf, lambda_param=2500.0, alpha=2.0 / 3.0)


recon = apply_spatially_varying_deconv(
    image=blurred_gray,
    psf_grid=cal["psf_grid"],
    positions=cal["positions"],
    deconv_fn=_deconv_fn,
    patch_size=192,
    overlap=0.5,
)
recon = np.clip(recon, 0.0, 1.0)

cv2.imwrite(
    str(STEP56_DIR / "blurred_gray.png"),
    (blurred_gray * 255.0).astype(np.uint8),
)
cv2.imwrite(
    str(STEP56_DIR / "reference_gray.png"),
    (reference_gray * 255.0).astype(np.uint8),
)
cv2.imwrite(
    str(STEP56_DIR / "deconvolved_gray.png"),
    (recon * 255.0).astype(np.uint8),
)
np.save(STEP56_DIR / "deconvolved_gray.npy", recon)

mse = float(np.mean((reference_gray - recon) ** 2))
psnr = float("inf") if mse <= np.finfo(float).eps else float(10.0 * np.log10(1.0 / mse))

summary = {
    "calibration": {
        "patch_count": len(cal["positions"]),
        "undistorted": bool(cal["undistorted"]),
        "psf_size": 31,
        "patch_size": 192,
        "method": "wiener",
        "per_channel": False,
        "output_npz": "calibration_triplet_a.npz",
    },
    "deconvolution": {
        "lambda_param": 2500.0,
        "alpha": 2.0 / 3.0,
        "mse_vs_reference": mse,
        "psnr_vs_reference": psnr,
        "output_image": "deconvolved_gray.png",
        "output_array": "deconvolved_gray.npy",
    },
}

(STEP56_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
