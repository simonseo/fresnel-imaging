import pytest

import fresnel_imaging as fresnel_imaging_pkg
from fresnel_imaging import simulation


class TestSimulationLazyImports:
    @pytest.mark.parametrize(
        "name",
        [
            "fresnel_lens",
            "lens_geometry",
            "materials",
            "psf_measurement",
            "render",
            "scene_setup",
        ],
    )
    def test_all_declared_submodules_lazy_import(self, name):
        module = getattr(simulation, name)
        assert module.__name__.endswith(f"simulation.{name}")

    def test_repeated_access_returns_same_module(self):
        m1 = simulation.lens_geometry
        m2 = simulation.lens_geometry
        assert m1 is m2

    @pytest.mark.parametrize(
        "name",
        [
            "fresnel_lens",
            "lens_geometry",
            "materials",
            "psf_measurement",
            "render",
            "scene_setup",
        ],
    )
    def test_repeated_access_returns_same_module_for_each_submodule(self, name):
        m1 = getattr(simulation, name)
        m2 = getattr(simulation, name)
        assert m1 is m2

    def test_from_import_submodule(self):
        from fresnel_imaging.simulation import lens_geometry

        assert lens_geometry.__name__.endswith("simulation.lens_geometry")

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = simulation.not_a_real_simulation_module


class TestTopLevelPackageLazyImport:
    def test_top_level_simulation_attribute_imports_once(self):
        module = fresnel_imaging_pkg.simulation
        assert module.__name__.endswith("fresnel_imaging.simulation")

    def test_top_level_simulation_repeated_access_cached(self):
        m1 = fresnel_imaging_pkg.simulation
        m2 = fresnel_imaging_pkg.simulation
        assert m1 is m2
