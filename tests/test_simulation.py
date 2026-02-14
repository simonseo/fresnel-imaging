from fresnel_imaging.simulation.fresnel_lens import compute_profile


class TestComputeProfile:
    def test_compute_profile_length(self):
        profile = compute_profile(diameter=50.0, focal_length=30.0, n_grooves=5, thickness=5.0)
        assert isinstance(profile, list)
        assert len(profile) > 0
        for point in profile:
            assert isinstance(point, tuple)
            assert len(point) == 2

    def test_compute_profile_radial_range(self):
        diameter = 50.0
        focal_length = 30.0
        n_grooves = 8
        thickness = 5.0
        profile = compute_profile(diameter, focal_length, n_grooves, thickness)
        radii = [r for r, z in profile]
        assert min(radii) >= 0.0
        assert max(radii) <= diameter / 2.0 + 0.1

    def test_compute_profile_groove_angles(self):
        diameter = 60.0
        focal_length = 40.0
        n_grooves = 10
        thickness = 6.0
        profile = compute_profile(diameter, focal_length, n_grooves, thickness)
        groove_heights = [z for r, z in profile[1:] if r > 0]
        if len(groove_heights) >= 2:
            for i in range(len(groove_heights) - 1):
                assert groove_heights[i] >= 0.0
