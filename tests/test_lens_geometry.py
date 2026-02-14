import numpy as np
import pytest

from fresnel_imaging.simulation.lens_geometry import (
    GLASS_CATALOG,
    _element_profile,
    _surface_sag,
    compute_planoconvex_profile,
    validate_prescription,
)
from fresnel_imaging.simulation.psf_measurement import extract_psf_from_render

# ── Plano-convex profile ───────────────────────────────────────────


class TestComputePlanoconvexProfile:
    def test_profile_returns_list_of_tuples(self):
        profile = compute_planoconvex_profile(50.0, 100.0, 5.0)
        assert isinstance(profile, list)
        assert len(profile) > 2
        for pt in profile:
            assert isinstance(pt, tuple)
            assert len(pt) == 2

    def test_profile_radial_range(self):
        diameter = 50.0
        profile = compute_planoconvex_profile(diameter, 100.0, 5.0)
        radii = [r for r, z in profile]
        assert min(radii) >= 0.0
        assert max(radii) <= diameter / 2.0 + 0.01

    def test_profile_z_range(self):
        """Edge of curved face should reach thickness; flat face at z=0."""
        profile = compute_planoconvex_profile(50.0, 100.0, 5.0)
        z_vals = [z for _r, z in profile]
        assert 0.0 in z_vals
        assert max(z_vals) == pytest.approx(5.0, abs=0.01)

    def test_profile_closed(self):
        """First and last points should be at r=0 (closed for revolution)."""
        profile = compute_planoconvex_profile(50.0, 100.0, 5.0)
        assert profile[0][0] == 0.0  # centre-top
        assert profile[-1][0] == 0.0  # centre-bottom

    def test_curved_face_monotonic(self):
        """Curved face z should increase monotonically with radius."""
        profile = compute_planoconvex_profile(50.0, 100.0, 5.0, n_points=64)
        curved = profile[:64]
        z_vals = [z for _r, z in curved]
        for i in range(len(z_vals) - 1):
            assert z_vals[i] <= z_vals[i + 1] + 1e-9

    def test_fast_lens_validation(self):
        """f/0.5 lens should raise ValueError (semi-diam > radius of curvature)."""
        with pytest.raises(ValueError, match="Semi-diameter"):
            compute_planoconvex_profile(
                diameter=200.0, focal_length=50.0, thickness=5.0, material_ior=1.5
            )

    def test_very_slow_lens(self):
        """f/20 lens should work fine."""
        profile = compute_planoconvex_profile(
            diameter=10.0, focal_length=200.0, thickness=2.0
        )
        assert len(profile) > 2

    def test_n_points_parameter(self):
        p16 = compute_planoconvex_profile(50.0, 100.0, 5.0, n_points=16)
        p64 = compute_planoconvex_profile(50.0, 100.0, 5.0, n_points=64)
        # More points → more profile entries (each adds 1 to curved face)
        assert len(p64) > len(p16)


# ── Glass catalog ──────────────────────────────────────────────────


class TestGlassCatalog:
    def test_catalog_has_expected_entries(self):
        expected = {"BK7", "SF5", "LAK9", "FUSED_SILICA", "PMMA"}
        assert expected.issubset(GLASS_CATALOG.keys())

    def test_catalog_values_are_tuples(self):
        for name, (ior, cauchy_b) in GLASS_CATALOG.items():
            assert isinstance(ior, float), f"{name}: IOR not float"
            assert isinstance(cauchy_b, float), f"{name}: Cauchy-B not float"
            assert ior > 1.0, f"{name}: IOR must be > 1"
            assert cauchy_b >= 0.0, f"{name}: Cauchy-B must be >= 0"


# ── Prescription validation ────────────────────────────────────────


class TestValidatePrescription:
    def test_missing_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            validate_prescription([{"radius": 100.0}])

    def test_negative_thickness(self):
        with pytest.raises(ValueError, match="thickness must be positive"):
            validate_prescription([
                {"radius": 100.0, "thickness": -5.0, "ior": 1.5, "diameter": 25.0},
                {"radius": -50.0, "thickness": 0.0, "ior": 1.0, "diameter": 25.0},
            ])

    def test_invalid_ior(self):
        with pytest.raises(ValueError, match="IOR must be >= 1.0"):
            validate_prescription([
                {"radius": 100.0, "thickness": 5.0, "ior": 0.5, "diameter": 25.0},
                {"radius": -50.0, "thickness": 0.0, "ior": 1.0, "diameter": 25.0},
            ])

    def test_valid_doublet(self):
        """A valid cemented doublet prescription should not raise."""
        prescription = [
            {"radius": 61.47, "thickness": 6.0, "ior": 1.5168, "diameter": 25.0},
            {"radius": -43.47, "thickness": 3.0, "ior": 1.6727, "diameter": 25.0},
            {"radius": -124.1, "thickness": 0.0, "ior": 1.0, "diameter": 25.0},
        ]
        validate_prescription(prescription)  # should not raise

    def test_empty_prescription(self):
        with pytest.raises(ValueError, match="at least one surface"):
            validate_prescription([])


class TestLensProfileHelpers:
    def test_surface_sag_flat_surface(self):
        assert _surface_sag(4.0, np.inf) == 0.0

    def test_surface_sag_positive_curvature(self):
        s0 = _surface_sag(0.0, 50.0)
        s1 = _surface_sag(5.0, 50.0)
        s2 = _surface_sag(10.0, 50.0)
        assert s0 == pytest.approx(0.0)
        assert 0.0 < s1 < s2

    def test_surface_sag_negative_curvature(self):
        s = _surface_sag(8.0, -60.0)
        assert s < 0.0

    def test_element_profile_length(self):
        profile = _element_profile(
            front_radius=60.0,
            back_radius=-45.0,
            thickness=5.0,
            diameter=20.0,
            n_points=32,
        )
        assert len(profile) == 64

    def test_element_profile_edge_and_center(self):
        profile = _element_profile(
            front_radius=80.0,
            back_radius=-80.0,
            thickness=6.0,
            diameter=30.0,
            n_points=16,
        )
        front_edge = profile[15]
        back_edge = profile[16]
        front_center = profile[0]
        back_center = profile[-1]

        assert front_edge[0] == pytest.approx(15.0)
        assert back_edge[0] == pytest.approx(15.0)
        assert front_edge[1] == pytest.approx(6.0, abs=1e-9)
        assert back_edge[1] == pytest.approx(0.0, abs=1e-9)
        assert front_center[0] == pytest.approx(0.0)
        assert back_center[0] == pytest.approx(0.0)


# ── PSF extraction ─────────────────────────────────────────────────


class TestExtractPsfFromRender:
    def test_centered_gaussian(self):
        y, x = np.mgrid[-32:32, -32:32]
        img = np.exp(-(x**2 + y**2) / (2 * 5**2))
        psf = extract_psf_from_render(img, psf_size=31)
        assert psf.shape == (31, 31)
        assert abs(psf.sum() - 1.0) < 1e-6
        assert psf.dtype == np.float64

    def test_off_center_gaussian(self):
        img = np.zeros((64, 64), dtype=np.float64)
        y, x = np.mgrid[0:11, 0:11]
        img[40:51, 45:56] = np.exp(-(((x - 5) ** 2 + (y - 5) ** 2)) / (2 * 2**2))
        psf = extract_psf_from_render(img, psf_size=11)
        assert psf.shape == (11, 11)
        assert abs(psf.sum() - 1.0) < 1e-6

    def test_multichannel_input(self):
        y, x = np.mgrid[-16:16, -16:16]
        gray = np.exp(-(x**2 + y**2) / (2 * 3**2))
        rgb = np.stack([gray, gray, gray], axis=-1)
        psf = extract_psf_from_render(rgb, psf_size=15)
        assert psf.shape == (15, 15)
        assert abs(psf.sum() - 1.0) < 1e-6

    def test_very_dim_image(self):
        """Near-zero image should return uniform PSF, not NaN/inf."""
        img = np.full((32, 32), 1e-12, dtype=np.float64)
        psf = extract_psf_from_render(img, psf_size=11, threshold=0.001)
        assert psf.shape == (11, 11)
        assert not np.any(np.isnan(psf))
        assert not np.any(np.isinf(psf))
        assert abs(psf.sum() - 1.0) < 1e-6

    def test_custom_psf_size(self):
        y, x = np.mgrid[-32:32, -32:32]
        img = np.exp(-(x**2 + y**2) / (2 * 5**2))
        for size in [7, 15, 21, 41]:
            psf = extract_psf_from_render(img, psf_size=size)
            assert psf.shape == (size, size)
            assert abs(psf.sum() - 1.0) < 1e-6

    def test_even_psf_size_supported(self):
        y, x = np.mgrid[-20:20, -20:20]
        img = np.exp(-(x**2 + y**2) / (2 * 4**2))
        psf = extract_psf_from_render(img, psf_size=20)
        assert psf.shape == (20, 20)
        assert abs(psf.sum() - 1.0) < 1e-6

    def test_edge_centroid_zero_padding(self):
        img = np.zeros((30, 30), dtype=np.float64)
        img[1, 2] = 100.0
        psf = extract_psf_from_render(img, psf_size=15)
        assert psf.shape == (15, 15)
        assert abs(psf.sum() - 1.0) < 1e-6
        assert np.count_nonzero(psf) == 1
        peak_row, peak_col = np.unravel_index(np.argmax(psf), psf.shape)
        assert peak_row <= 8
        assert peak_col <= 8

    def test_negative_values_clipped(self):
        img = np.zeros((33, 33), dtype=np.float64)
        img[16, 16] = 10.0
        img[16, 17] = -5.0
        psf = extract_psf_from_render(img, psf_size=9)
        assert np.all(psf >= 0.0)
        assert abs(psf.sum() - 1.0) < 1e-6

    def test_single_channel_3d_input(self):
        img = np.zeros((21, 21, 1), dtype=np.float64)
        img[10, 10, 0] = 1.0
        psf = extract_psf_from_render(img, psf_size=11)
        assert psf.shape == (11, 11)
        assert abs(psf.sum() - 1.0) < 1e-6

    def test_missing_image_path_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_psf_from_render("/tmp/definitely_missing_psf_image_12345.exr")
