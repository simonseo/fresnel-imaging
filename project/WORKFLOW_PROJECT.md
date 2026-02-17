# Project Workflow Runbook

This runbook executes steps 1-8 of the Fresnel workflow using the `project/`
workspace artifacts.

## 1) Toolchain checks

```bash
"/Applications/Blender.app/Contents/MacOS/Blender" -b -P tests/manual/blender_check.py
```

Optional BlendLuxCore verification (headless):

```bash
python3 - <<'PY'
script='''
import bpy
try:
    bpy.context.scene.render.engine = "LUXCORE"
    print("LUXCORE available")
except Exception as exc:
    print("LUXCORE unavailable:", exc)
'''
open('/tmp/check_luxcore_engine.py','w').write(script)
PY
"/Applications/Blender.app/Contents/MacOS/Blender" -b -P /tmp/check_luxcore_engine.py
```

## 2-3) Lens geometry + scene/camera/material setup

```bash
"/Applications/Blender.app/Contents/MacOS/Blender" -b -P project/code/generate_lens_assets.py
```

Outputs:
- `project/assets/common/` (modular Cornell box)
- `project/assets/lenses/` (six lens packs with scene + models + previews)

## 4) Triplet step-4 renders + PSF extraction

```bash
"/Applications/Blender.app/Contents/MacOS/Blender" -b -P project/code/triplet_step4_render.py
PYTHONPATH=src python3 project/code/triplet_step4_extract_psf.py
```

Outputs under `project/assets/lenses/triplet_a/step4/` include:
- `reference_scene.png`
- `blurred_scene.exr`, `blurred_scene.png`
- `psf_point_center.png`, `psf_point_offaxis.png`
- `psf_center.npy`, `psf_offaxis.npy`, `psf_summary.json`

## 5-6) Calibration + deconvolution (triplet)

```bash
PYTHONPATH=src python3 project/code/triplet_step56_calibrate_deconv.py
```

Outputs under `project/assets/lenses/triplet_a/step5_step6/` include:
- `calibration_triplet_a.npz`
- `deconvolved_gray.png`, `deconvolved_gray.npy`
- `summary.json`

## 7) Operational split

- Blender scripts: scene/lens simulation and rendered captures.
- Python scripts (`PYTHONPATH=src`): PSF extraction, calibration, deconvolution.

## 8) Reproducibility verification

```bash
python3 project/code/validate_project_assets.py
```

Expected output: `Project assets validation passed`
