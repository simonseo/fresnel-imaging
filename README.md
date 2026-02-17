# Fresnel Lens Computational Imaging

Python pipeline for computational imaging through Fresnel lenses, based on the research paper
**"Computational Imaging with Fresnel Lenses"** by Simon Myunggun Seo (Carnegie Mellon University).

Fresnel lenses are thin, lightweight, and cheap — but introduce severe optical aberrations.
This package computationally reverses those aberrations through deconvolution, making Fresnel
lenses viable for practical imaging.

Repository: <https://github.com/simonseo/fresnel-imaging>

This project documentation and implementation were created with help from OpenCode.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[gpu]"   # PyTorch GPU acceleration
pip install -e ".[dev]"   # pytest, ruff
```

Simulation modules require [Blender](https://www.blender.org/) with the
[LuxCoreRender](https://luxcorerender.org/) addon.

Tested/latest compatible versions for this project:
- Blender `4.5.6` (LTS macOS ARM build: `blender-4.5.6-macos-arm64.dmg`)
- BlendLuxCore `2.10.2` addon zip (`BlendLuxCore-2.10.2.zip`)

Download links:
- Blender LTS 4.5: <https://www.blender.org/download/lts/4-5/>
- BlendLuxCore releases (download the addon zip): <https://github.com/LuxCoreRender/BlendLuxCore/releases>

## Quick Start

### CLI

```bash
fresnel-deblur --image blurred.jpg --kernel psf.png --method all --output-dir results/
```

Methods: `pd` (primal-dual cross-channel), `hyper` (hyper-Laplacian), `yuv` (YUV multi-channel), `all`.

### Python API

**Deconvolution:**

```python
import numpy as np
from PIL import Image
from fresnel_imaging.deconvolution import fast_deconv, fast_deconv_yuv

img = np.array(Image.open("blurred.jpg")).astype(np.float64) / 255.0
kernel = np.array(Image.open("psf.png").convert("L")).astype(np.float64)
kernel /= kernel.sum()

restored = fast_deconv(img[:, :, 0], kernel, lambda_param=2000, alpha=2/3)
restored_color = fast_deconv_yuv(img, kernel, lambda_param=2e3, alpha=0.65)
```

**PSF Calibration:**

```python
from fresnel_imaging.calibration import calibrate_fresnel_lens, save_calibration

result = calibrate_fresnel_lens(
    blurred_image=blurred,
    reference_image=sharp_reference,
    psf_size=31,
    patch_size=64,
    method="qp",
)

save_calibration("calibration.npz",
    camera_matrix=result["camera_matrix"],
    dist_coeffs=result["dist_coeffs"],
    psf_grid=result["psf_grid"],
    positions=result["positions"],
)
```

**Simulation (inside Blender):**

```python
from fresnel_imaging.simulation import fresnel_lens, materials, scene_setup, render

lens = fresnel_lens.create_fresnel_lens(diameter=100, focal_length=200, n_grooves=50)
materials.assign_material(lens, materials.create_glass_material())
scene_setup.setup_imaging_scene(lens, image_path="target.jpg", object_distance=500)
render.render_scene("output.exr")
```

## Package Structure

```
src/fresnel_imaging/
├── deconvolution/          # MATLAB-to-Python port of deconvolution algorithms
│   ├── utils.py            # psf2otf, edgetaper, imconv, boundary handling
│   ├── solve_image.py      # Proximal operator for hyper-Laplacian priors
│   ├── pd_joint_deconv.py  # Primal-dual cross-channel (Chambolle-Pock)
│   ├── fast_deconv.py      # Half-quadratic splitting (Krishnan & Fergus 2009)
│   ├── fast_deconv_yuv.py  # YUV multi-channel (Schuler et al. 2011)
│   └── metrics.py          # SNR, PSNR, SSIM
├── simulation/             # Blender-based optical simulation
│   ├── fresnel_lens.py     # Procedural geometry via bmesh (Snell's law groove angles)
│   ├── lens_geometry.py    # Plano-convex and multi-element lens builders
│   ├── materials.py        # LuxCoreRender/Cycles glass materials
│   ├── psf_measurement.py  # PSF measurement via point source rendering
│   ├── scene_setup.py      # Calibration and imaging scene configuration
│   └── render.py           # Headless rendering automation
├── calibration/            # PSF estimation pipeline
│   ├── distortion.py       # Lens distortion correction (OpenCV)
│   ├── alignment.py        # SIFT + RANSAC alignment (global + patchwise)
│   ├── psf_estimation.py   # QP, Wiener, Richardson-Lucy, spatially-varying PSF
│   └── pipeline.py         # End-to-end calibration with save/load
└── cli.py                  # fresnel-deblur command-line interface
```

## Workflow Guide

For a comprehensive step-by-step guide covering lens modeling, PSF measurement,
calibration, and deconvolution, see [WORKFLOW.md](WORKFLOW.md).

## Attribution and Sources

- Canonical repository: <https://github.com/simonseo/fresnel-imaging>
- Created with help from OpenCode.
- Algorithmic foundations: Seo (CMU), Heide et al. (SIGGRAPH 2013), Krishnan & Fergus (NIPS 2009), Schuler et al. (ICCV 2011), Mannan & Langer (CRV 2016).
- Deconvolution and calibration components are Python reimplementations guided by the cited papers and prior MATLAB pipeline behavior; they are not direct verbatim copies of paper text.
- Third-party dependencies (NumPy, SciPy, OpenCV, Blender, LuxCore) remain under their own upstream licenses.

## References

- Seo, "Computational Imaging with Fresnel Lenses" (CMU)
- Heide et al., "High-Quality Computational Imaging Through Simple Lenses" (SIGGRAPH 2013)
- Krishnan & Fergus, "Fast Image Deconvolution using Hyper-Laplacian Priors" (NIPS 2009)
- Schuler et al., "Non-stationary Correction of Optical Aberrations" (ICCV 2011)
- Mannan & Langer, "Blur Calibration for Depth from Defocus" (CRV 2016)
