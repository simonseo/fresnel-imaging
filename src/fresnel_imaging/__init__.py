"""Fresnel lens computational imaging pipeline."""

from importlib import import_module

from fresnel_imaging import calibration, deconvolution

__all__ = [
    "calibration",
    "deconvolution",
    "simulation",
]


def __getattr__(name: str):  # noqa: N807
    if name == "simulation":
        module = import_module(".simulation", __name__)
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
