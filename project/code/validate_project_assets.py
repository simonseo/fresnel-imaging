import json
from pathlib import Path

import numpy as np

ROOT = Path("/Users/sseo/Documents/fresnel-imaging")
PROJECT = ROOT / "project"
COMMON = PROJECT / "assets" / "common"
LENSES = PROJECT / "assets" / "lenses"


def _assert_files(base: Path, names: list[str]) -> None:
    missing = [name for name in names if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {base}: {missing}")


def main() -> None:
    _assert_files(
        COMMON,
        ["cornell_box.blend", "cornell_box.glb", "cornell_box_preview.png"],
    )

    lens_ids = sorted(path.name for path in LENSES.iterdir() if path.is_dir())
    expected = [
        "biconvex_a",
        "biconvex_b",
        "fresnel_a",
        "fresnel_b",
        "fresnel_c",
        "triplet_a",
    ]
    if lens_ids != expected:
        raise ValueError(f"Unexpected lens ids: {lens_ids}")

    for lens_id in lens_ids:
        base = LENSES / lens_id
        _assert_files(
            base,
            [
                "metadata.json",
                "scene.blend",
                "preview.png",
                "scene_preview.png",
                "lens.glb",
                "lens.gltf",
                "lens.bin",
            ],
        )

    triplet = LENSES / "triplet_a"
    step4 = triplet / "step4"
    step56 = triplet / "step5_step6"
    _assert_files(
        step4,
        [
            "reference_scene.png",
            "blurred_scene.exr",
            "blurred_scene.png",
            "psf_point_center.png",
            "psf_point_offaxis.png",
            "psf_center.npy",
            "psf_offaxis.npy",
            "psf_summary.json",
        ],
    )
    _assert_files(
        step56,
        [
            "calibration_triplet_a.npz",
            "deconvolved_gray.png",
            "deconvolved_gray.npy",
            "summary.json",
        ],
    )

    psf_center = np.load(step4 / "psf_center.npy")
    psf_offaxis = np.load(step4 / "psf_offaxis.npy")
    if not np.isclose(psf_center.sum(), 1.0, atol=1e-6):
        raise ValueError("Center PSF is not normalized")
    if not np.isclose(psf_offaxis.sum(), 1.0, atol=1e-6):
        raise ValueError("Offaxis PSF is not normalized")

    triplet_meta = json.loads((triplet / "metadata.json").read_text(encoding="utf-8"))
    if "step4" not in triplet_meta.get("files", {}):
        raise ValueError("triplet metadata missing step4 file references")
    if "step5_step6" not in triplet_meta.get("files", {}):
        raise ValueError("triplet metadata missing step5_step6 references")

    print("Project assets validation passed")


if __name__ == "__main__":
    main()
