import tempfile
from pathlib import Path

import numpy as np

from fresnel_imaging.calibration.alignment import extract_patches, stitch_patches
from fresnel_imaging.calibration.pipeline import load_calibration, save_calibration
from fresnel_imaging.calibration.psf_estimation import (
    estimate_psf_rl,
    estimate_psf_wiener,
    interpolate_psf,
)


class TestEstimatePSFWiener:
    def test_estimate_psf_wiener_shape(self):
        blurred = np.random.rand(64, 64).astype(np.float32)
        sharp = np.random.rand(64, 64).astype(np.float32)
        psf = estimate_psf_wiener(blurred, sharp, noise_var=0.01)
        assert psf.ndim == 2
        assert psf.shape[0] == psf.shape[1]

    def test_estimate_psf_wiener_sums_to_one(self):
        blurred = np.random.rand(64, 64).astype(np.float32)
        sharp = np.random.rand(64, 64).astype(np.float32)
        psf = estimate_psf_wiener(blurred, sharp, noise_var=0.01)
        assert np.isclose(psf.sum(), 1.0, atol=1e-6)


class TestEstimatePSFRL:
    def test_estimate_psf_rl_runs(self):
        blurred = np.random.rand(64, 64).astype(np.float32)
        sharp = np.random.rand(64, 64).astype(np.float32)
        psf = estimate_psf_rl(blurred, sharp, n_iter=10)
        assert psf.shape == (31, 31)
        assert psf.dtype == np.float64
        assert not np.any(np.isnan(psf))


class TestExtractPatches:
    def test_extract_patches_covers_image(self):
        image = np.random.rand(128, 96)
        patches, origins = extract_patches(image, patch_size=32, overlap=0.5)
        assert len(patches) > 0
        assert len(patches) == len(origins)
        max_row = max(origins, key=lambda x: x[0])[0]
        max_col = max(origins, key=lambda x: x[1])[1]
        assert max_row + patches[-1].shape[0] >= image.shape[0] - 1
        assert max_col + patches[-1].shape[1] >= image.shape[1] - 1


class TestStitchPatches:
    def test_stitch_patches_roundtrip(self):
        image = np.random.rand(64, 64)
        patches, origins = extract_patches(image, patch_size=32, overlap=0.5)
        stitched = stitch_patches(patches, origins, image.shape, overlap=0.5)
        assert stitched.shape == image.shape
        assert not np.any(np.isnan(stitched))


class TestInterpolatePSF:
    def test_interpolate_psf_identity(self):
        psf_grid = [np.ones((5, 5)) / 25.0]
        positions = [(32, 32)]
        query = (32.0, 32.0)
        result = interpolate_psf(psf_grid, positions, query)
        assert result.shape == (5, 5)
        assert np.allclose(result, psf_grid[0])


class TestSaveLoadCalibration:
    def test_save_load_calibration_roundtrip(self):
        camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.1, -0.05, 0.001, 0.002])
        psf_grid = [
            np.ones((7, 7)) / 49.0,
            np.ones((7, 7)) / 49.0,
        ]
        positions = [(32, 32), (96, 96)]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "calibration.npz"
            save_calibration(
                filepath,
                camera_matrix,
                dist_coeffs,
                psf_grid,
                positions,
            )
            result = load_calibration(filepath)

            assert result["camera_matrix"] is not None
            assert np.allclose(result["camera_matrix"], camera_matrix)
            assert result["dist_coeffs"] is not None
            assert np.allclose(result["dist_coeffs"], dist_coeffs)
            assert len(result["psf_grid"]) == len(psf_grid)
            assert result["positions"] == positions
