import numpy as np

from fresnel_imaging.deconvolution.fast_deconv import fast_deconv
from fresnel_imaging.deconvolution.fast_deconv_yuv import fast_deconv_yuv
from fresnel_imaging.deconvolution.metrics import psnr, snr
from fresnel_imaging.deconvolution.solve_image import solve_image
from fresnel_imaging.deconvolution.utils import (
    edgetaper,
    imconv,
    img_to_norm_grayscale,
    psf2otf,
)


class TestPSF2OTF:
    def test_psf2otf_identity(self):
        psf = np.zeros((3, 3))
        psf[1, 1] = 1.0
        otf = psf2otf(psf, (5, 5))
        assert otf.shape == (5, 5)
        assert np.allclose(np.abs(otf), 1.0)

    def test_psf2otf_shape(self):
        psf = np.ones((3, 3))
        otf = psf2otf(psf, (8, 8))
        assert otf.shape == (8, 8)
        assert otf.dtype == np.complex128

    def test_psf2otf_zero(self):
        psf = np.zeros((3, 3))
        otf = psf2otf(psf, (5, 5))
        assert np.allclose(otf, 0.0)


class TestEdgetaper:
    def test_edgetaper_preserves_center(self):
        image = np.random.rand(32, 32)
        psf = np.ones((5, 5)) / 25.0
        result = edgetaper(image, psf)
        assert result.shape == image.shape
        center_crop = 5
        assert np.allclose(
            result[center_crop:-center_crop, center_crop:-center_crop],
            image[center_crop:-center_crop, center_crop:-center_crop],
            atol=1e-6,
        )

    def test_edgetaper_output_shape(self):
        image = np.random.rand(64, 48)
        psf = np.ones((7, 7)) / 49.0
        result = edgetaper(image, psf)
        assert result.shape == image.shape


class TestImconv:
    def test_imconv_identity(self):
        image = np.random.rand(16, 16)
        kernel = np.array([[1.0]])
        result = imconv(image, kernel, output="same")
        assert np.allclose(result, image, atol=1e-10)

    def test_imconv_same_shape(self):
        image = np.random.rand(20, 20)
        kernel = np.ones((3, 3)) / 9.0
        result = imconv(image, kernel, output="same")
        assert result.shape == image.shape

    def test_imconv_full_shape(self):
        image = np.random.rand(10, 10)
        kernel = np.ones((3, 3))
        result = imconv(image, kernel, output="full")
        assert result.shape == (12, 12)


class TestSolveImageSoftThreshold:
    def test_solve_image_soft_threshold(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        beta = 1.0
        alpha = 1.0
        result = solve_image(v, beta, alpha)
        assert result.shape == v.shape
        expected = np.array([[0.0, 1.0], [2.0, 3.0]])
        assert np.allclose(result, expected, atol=1e-6)


class TestFastDeconv:
    def test_fast_deconv_runs(self):
        image = np.random.rand(64, 64)
        kernel = np.ones((5, 5)) / 25.0
        result = fast_deconv(image, kernel, lambda_param=0.1, alpha=0.5)
        assert result.shape == image.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestFastDeconvYUV:
    def test_fast_deconv_yuv_runs(self):
        image = np.random.rand(64, 64, 3)
        kernels = [
            np.ones((5, 5)) / 25.0,
            np.ones((5, 5)) / 25.0,
            np.ones((5, 5)) / 25.0,
        ]
        result = fast_deconv_yuv(
            image,
            kernels,
            lambda_param=0.1,
            rho_yuv=[0.1, 0.1, 0.1],
            w_rgb=[1.0, 1.0, 1.0],
            theta=0.1,
            alpha=0.5,
        )
        assert result.shape[2] == 3
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestPSNR:
    def test_psnr_identical(self):
        image = np.random.rand(32, 32)
        psnr_val = psnr(image, image)
        assert np.isinf(psnr_val) or psnr_val > 50.0

    def test_psnr_noisy(self):
        image = np.random.rand(32, 32)
        noisy = image + 0.01 * np.random.randn(32, 32)
        noisy = np.clip(noisy, 0, 1)
        psnr_val = psnr(noisy, image)
        assert psnr_val > 0
        assert not np.isnan(psnr_val)


class TestSNR:
    def test_snr_returns_tuple(self):
        image = np.random.rand(64, 64)
        noisy = image + 0.01 * np.random.randn(64, 64)
        noisy = np.clip(noisy, 0, 1)
        result = snr(noisy, 2, image)
        assert isinstance(result, tuple)
        assert len(result) == 2
        snr_val, mse_val = result
        assert isinstance(snr_val, float)
        assert isinstance(mse_val, float)
        assert snr_val > 0
        assert mse_val >= 0


class TestImgToNormGrayscale:
    def test_img_to_norm_grayscale_rgb_uint8(self):
        img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        result = img_to_norm_grayscale(img)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_img_to_norm_grayscale_grayscale(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        result = img_to_norm_grayscale(img)
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)
        assert len(result) == 3

    def test_img_to_norm_grayscale_float(self):
        img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = img_to_norm_grayscale(img)
        assert result.dtype == np.float64
        assert np.allclose(result, img, atol=1e-6)
