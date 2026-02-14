# Fresnel Lens Computational Imaging — Complete Workflow Tutorial

Build lens models in Blender, render physically accurate scenes, measure
PSFs, and computationally reverse optical aberrations.  No prior knowledge
of optics, Blender, or signal processing is assumed.

---

## 1. Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Runtime for all scripts |
| **Blender** | 3.6 LTS or 4.x | 3-D scene and mesh engine |
| **BlendLuxCore** | 2.7+ | LuxCoreRender addon (glass caustics) |
| **OpenCV** | 4.x | Image I/O, distortion correction |
| **NumPy / SciPy** | latest | Array math and optimisation |

```bash
git clone https://github.com/simonseo/fresnel-imaging.git && cd fresnel-imaging
pip install -e .            # core package
pip install -e ".[gpu]"     # (optional) GPU acceleration
pip install -e ".[dev]"     # (optional) pytest, ruff
```

Canonical repository: <https://github.com/simonseo/fresnel-imaging>

This tutorial and implementation were created with help from OpenCode.

**Blender setup:**
1. Download from <https://www.blender.org/download/>.
2. Install BlendLuxCore addon zip via **Edit → Preferences → Add-ons → Install…**.
3. Ensure `blender` is on `$PATH` or pass `blender_bin=` explicitly.

> Every Blender command in this tutorial uses `blender -b -P <script.py>`
> (the `-b` flag = headless).  You never need the GUI.

---

## 2. Optical Concepts Primer

Skip to §3 if you already know thin-lens optics and deconvolution.

### 2.1 Point Spread Function (PSF)

A perfect lens images a point source as a perfect dot.  A real lens
spreads it into a blob — the **PSF**.

```
  Point Source             Perfect Lens              Fresnel Lens
       *            ──────►      .           ──────►    . : . .
                                                        : * : .
                                                        . : . .
```

The PSF characterises the blur at a given field position.  If you know
it, you can reverse the blur via **deconvolution**.

### 2.2 Thin Lens Equation

```
     1       1       1
    ───  =  ───  +  ───        →    dᵢ  =  1 / (1/f − 1/d₀)
     f       d₀      dᵢ
```

`_thin_lens_image_distance(focal_length, object_distance)` implements
this.  Raises `ValueError` when `d₀ ≤ f` (no real image).

### 2.3 Coordinate System

```
                     Optical Axis  (Z)
    ◄───────────────────────────────────────────────────────►

    Object Plane        Lens (origin)         Sensor / Camera
    z = −d₀             z = 0                  z = +dᵢ

    ┌──────────┐       /         \            ┌───────────┐
    │  scene   │      │  Fresnel  │           │  virtual  │
    │  or      │      │  or       │           │  sensor   │
    │  target  │      │  convex   │           │  plane    │
    └──────────┘       \         /            └───────────┘

         ─ d₀ ─►  ◄── 0 ──►  ◄──── dᵢ ────►
```

Objects at **−Z**, camera at **+Z**.  All distances in **millimetres**.

### 2.4 Lensmaker's Equation (Plano-Convex)

```
    1/f = (n − 1) / R   →   R = f · (n − 1)
    sag(r) = R − sqrt(R² − r²)
```

### 2.5 Fresnel Groove Physics

Each groove at radius *r* deflects a ray by `δ = arctan(r / f)`.
Snell's law gives the groove face angle:

```
    n · sin(α) = sin(α + δ)   →   α = arctan( sin(δ) / (n − cos(δ)) )
```

`_groove_face_angle(r, f, n)` implements this.

### 2.6 Deconvolution

A blurred image *y = K ⊛ x + noise*.  Deconvolution recovers *x*:

```
    min_x   (λ/2) ‖K ⊛ x − y‖²   +   Σ |∇ᵢ x|^α
```

Data fidelity (first term) + hyper-Laplacian prior (second term).
- `fast_deconv` — single-channel (Krishnan & Fergus 2009)
- `fast_deconv_yuv` — multi-channel YUV (Schuler et al. 2011)

---

## 3. Building a Lens Model

All lens construction functions live in the `simulation` subpackage and
must be called **inside Blender** (via `blender -b -P`).

### 3.1 Plano-Convex Lens — Worked Example

Create `build_planoconvex.py`:

```python
"""Build a plano-convex lens and save to .blend file."""
import bpy
from fresnel_imaging.simulation.lens_geometry import create_planoconvex_lens
from fresnel_imaging.simulation.materials import create_glass_material, assign_material

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

lens = create_planoconvex_lens(
    diameter=50.0,          # 50 mm clear aperture
    focal_length=100.0,     # 100 mm focal length
    thickness=5.0,          # 5 mm centre thickness
    material_ior=1.5168,    # BK7 glass
    name="PlanoConvex_BK7",
    segments=128,           # angular resolution
)

glass = create_glass_material(
    name="BK7_Glass",
    ior=1.5168,
    cauchy_b=0.00420,       # BK7 Cauchy-B dispersion
    transmission_color=(1.0, 1.0, 1.0),
    reflection_color=(1.0, 1.0, 1.0),
)
assign_material(lens, glass)

bpy.ops.wm.save_as_mainfile(filepath="planoconvex_bk7.blend")
print(f"Saved: {lens.name} — {len(lens.data.polygons)} faces")
```

```bash
blender -b -P build_planoconvex.py
```

**Under the hood:**
1. `compute_planoconvex_profile` → 2-D `(r, z)` cross-section via Lensmaker's eq.
2. `_revolve_profile` → 360° spin via `bmesh.ops.spin` → watertight mesh.
3. `create_glass_material` → LuxCore glass node (with Cauchy-B) or Cycles Glass BSDF fallback.
4. `assign_material` → replaces all materials on the object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `diameter` | float | Full lens diameter (mm) |
| `focal_length` | float | Paraxial focal length (mm) |
| `thickness` | float | Centre thickness (mm) |
| `material_ior` | float | Refractive index at 587.6 nm |
| `name` | str | Blender object name |
| `segments` | int | Angular segments (higher = smoother) |

> Semi-diameter must be < radius of curvature (`R = f·(n−1)`), else
> `compute_planoconvex_profile` raises `ValueError`.

### 3.2 Fresnel Lens

```python
"""Build a Fresnel lens."""
import bpy
from fresnel_imaging.simulation.fresnel_lens import create_fresnel_lens
from fresnel_imaging.simulation.materials import create_glass_material, assign_material

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

lens = create_fresnel_lens(
    diameter=100.0,         # 100 mm aperture
    focal_length=200.0,     # 200 mm focal length
    n_grooves=50,           # 50 concentric grooves
    thickness=2.0,          # 2 mm total (base + grooves)
    material_ior=1.4917,    # PMMA (acrylic)
    name="FresnelLens",
    segments=128,
)
glass = create_glass_material(name="PMMA_Glass", ior=1.4917, cauchy_b=0.0)
assign_material(lens, glass)
bpy.ops.wm.save_as_mainfile(filepath="fresnel_lens.blend")
```

```bash
blender -b -P build_fresnel.py
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `diameter` | 100.0 | Larger → more grooves needed |
| `focal_length` | 200.0 | Shorter → steeper groove angles |
| `n_grooves` | 50 | More → finer structure, more vertices |
| `thickness` | 2.0 | Base plate is 30% of total |
| `material_ior` | 1.5168 | See Appendix A |
| `segments` | 128 | 128–256 typical |

Profile computed by `compute_profile()` → calls `_groove_face_angle(r, f, n)`
per groove.

### 3.3 Multi-Element Lens from Prescription

Zemax-style prescription: each surface is a dict with `radius`,
`thickness`, `ior`, `diameter`.  Consecutive surfaces with `ior > 1.0`
form a single glass element.

```python
"""Build a cemented doublet from prescription data."""
import bpy
from fresnel_imaging.simulation.lens_geometry import (
    create_lens_from_prescription, validate_prescription,
)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

prescription = [
    {"radius": 61.47,  "thickness": 6.0, "ior": 1.5168, "diameter": 50.0},
    {"radius": -44.64, "thickness": 2.5, "ior": 1.0,    "diameter": 50.0},
    {"radius": -41.17, "thickness": 3.0, "ior": 1.6727, "diameter": 46.0},
    {"radius": -258.4, "thickness": 0.0, "ior": 1.0,    "diameter": 46.0},
]

validate_prescription(prescription)

elements = create_lens_from_prescription(
    prescription=prescription, name="Doublet", segments=128,
)

for i, elem in enumerate(elements):
    print(f"  Element {i}: {elem.name} at z={elem.location.z:.2f}")
bpy.ops.wm.save_as_mainfile(filepath="doublet.blend")
```

```bash
blender -b -P build_doublet.py
```

`create_lens_from_prescription` validates via `validate_prescription`,
builds each element with `_element_profile` → `_revolve_profile`,
positions along Z with cumulative offsets, assigns glass materials
from `GLASS_CATALOG` (IOR matched within ±0.002), and stores the full
prescription as JSON custom property.

### 3.4 General Guidance

- `segments=128` is adequate; use 256 for publication quality.
- Always use `create_glass_material` for optical elements.
- All dimensions in millimetres.
- Give each lens a unique `name` to avoid Blender auto-suffixing.
- Lens objects carry metadata as custom properties
  (`lens_focal_length`, `fresnel_focal_length`, etc.).

---

## 4. Setting Up Scenes

Three scene-setup functions place objects, cameras, and lights in
physically correct positions along the optical axis.

### 4.1 Calibration Scene (Checkerboard)

Used for distortion correction and PSF estimation from real or
simulated images.

```python
"""Set up a calibration scene with a checkerboard target."""
import bpy

from fresnel_imaging.simulation.fresnel_lens import create_fresnel_lens
from fresnel_imaging.simulation.materials import create_glass_material, assign_material
from fresnel_imaging.simulation.scene_setup import setup_calibration_scene

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Build the lens
lens = create_fresnel_lens(diameter=100.0, focal_length=200.0, n_grooves=50)
glass = create_glass_material(name="PMMA", ior=1.4917)
assign_material(lens, glass)

# Set up the calibration scene
result = setup_calibration_scene(
    lens_obj=lens,
    pattern_distance=500.0,     # checkerboard 500 mm in front
    sensor_distance=None,       # auto-compute via thin lens equation
    rows=8,                     # 8×8 grid
    cols=8,
    square_size=25.0,           # each square is 25 mm
)

print(f"Checkerboard at z = {result['checkerboard'].location.z:.1f}")
print(f"Camera at z = {result['image_distance']:.2f}")

bpy.ops.wm.save_as_mainfile(filepath="calibration_scene.blend")
```

```bash
blender -b -P setup_calibration.py
```

`setup_calibration_scene` returns a dict with keys:
- `"checkerboard"` — the Blender object (alternating B&W materials)
- `"camera"` — the camera object aimed at the origin
- `"image_distance"` — the computed sensor distance (mm)

### 4.2 Imaging Scene (Textured Plane)

Used for rendering a subject (photograph) through the lens.

```python
"""Set up an imaging scene with a photograph as the subject."""
import bpy

from fresnel_imaging.simulation.fresnel_lens import create_fresnel_lens
from fresnel_imaging.simulation.materials import create_glass_material, assign_material
from fresnel_imaging.simulation.scene_setup import setup_imaging_scene

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

lens = create_fresnel_lens(diameter=100.0, focal_length=200.0, n_grooves=50)
glass = create_glass_material(name="PMMA", ior=1.4917)
assign_material(lens, glass)

result = setup_imaging_scene(
    lens_obj=lens,
    image_path="target_photo.jpg",  # path to your subject image
    distance=1000.0,                # 1 m from lens
    sensor_distance=None,           # auto-compute
    plane_size=None,                # auto-scale from aspect ratio
)

print(f"Image plane at z = {result['image_plane'].location.z:.1f}")
print(f"Camera at z = {result['image_distance']:.2f}")

bpy.ops.wm.save_as_mainfile(filepath="imaging_scene.blend")
```

```bash
blender -b -P setup_imaging.py
```

The image is applied as an **emission shader** so it acts as a
self-illuminating source — no additional scene lighting is required.

`setup_imaging_scene` returns:
- `"image_plane"` — the textured plane object
- `"camera"` — the camera object
- `"image_distance"` — the computed sensor distance (mm)

### 4.3 PSF Measurement Scene

Used for measuring the lens's point spread function.  See §6 for the
full PSF workflow; this is a brief preview.

```python
from fresnel_imaging.simulation.psf_measurement import setup_psf_scene

result = setup_psf_scene(
    lens_obj=lens,
    point_position=(0.0, 0.0, -500.0),  # on-axis, 500 mm away
    sensor_distance=None,                # auto-compute
    render_samples=5000,
    render_resolution=(512, 512),
)
```

Returns: `{"point_source", "camera", "sensor_distance", "render_config"}`

---

## 5. Rendering

### 5.1 Render Engines

| Engine | Pros | Cons |
|--------|------|------|
| **LuxCore BIDIR** | Physically accurate caustics through glass | Slower, requires addon |
| **LuxCore PATH** | Faster than BIDIR for simple scenes | Weaker caustics |
| **Cycles** | Ships with Blender, no addon needed | Poor glass caustics |

**Recommendation:** Use LuxCore BIDIR for PSF measurement and any scene
involving light refracting through a lens.  Fall back to Cycles only if
LuxCore is not installed.

### 5.2 Configuring the Renderer

```python
from fresnel_imaging.simulation.render import configure_luxcore_render

configure_luxcore_render(
    samples=2000,           # halt after 2000 samples/pixel
    engine="BIDIR",         # bidirectional path tracing
    resolution=(1920, 1080),
    light_depth=15,         # max light-path bounces
    path_depth=15,          # max camera-path bounces
    sampler="METROPOLIS",   # good for caustics
    use_photongi=False,     # PhotonGI only for PATH engine
)
```

If LuxCore is unavailable, `configure_luxcore_render` automatically
falls back to `_configure_cycles_fallback` with equivalent Cycles
settings (caustics enabled, high bounce limits).

### 5.3 Rendering to Disk

```python
from fresnel_imaging.simulation.render import render_scene

# Render the current scene to a file
output_path = render_scene("output/render.exr")
print(f"Rendered to {output_path}")
```

The file format is inferred from the extension (`.exr` → OPEN_EXR,
`.png` → PNG, etc.) via `_format_from_ext`.  **Always use EXR for PSF
measurement** — PNG clips the dynamic range.

### 5.4 Headless Rendering

The `run_headless` function builds the command line; `execute_headless`
actually runs it:

```python
from fresnel_imaging.simulation.render import run_headless, execute_headless

# Option A: get the command line (list of strings)
cmd = run_headless(
    script_path="render_psf.py",
    blend_file=None,            # or "scene.blend"
    blender_bin=None,           # auto-detect
    extra_args=["--samples", "5000"],
)
print(" ".join(cmd))
# blender -b -P /abs/path/render_psf.py -- --samples 5000

# Option B: execute directly
result = execute_headless(
    script_path="render_psf.py",
    timeout=3600,  # 1 hour max
)
print(result.stdout)
```

Or simply call Blender from the shell:

```bash
blender -b -P render_psf.py -- --samples 5000
```

### 5.5 Batch Rendering

```python
from fresnel_imaging.simulation.render import batch_render, RenderConfig

configs = [
    RenderConfig(output_name="low_spp.exr",  samples=500,  engine="BIDIR"),
    RenderConfig(output_name="high_spp.exr", samples=5000, engine="BIDIR"),
]

paths = batch_render(
    blend_file="scene.blend",
    output_dir="output/batch/",
    configs=configs,
)
```

---

## 6. Measuring the PSF

The PSF is the fundamental calibration artefact.  This section covers
on-axis, off-axis, and spatially-varying PSF measurement.

### 6.1 On-Axis PSF (Single Point Source)

```
    Point Source          Fresnel Lens          Sensor (camera)
       (*)        ─────►  |||||||||  ─────►     [PSF blob]
    z = −500              z = 0                 z = +dᵢ
```

#### Full script: `measure_psf.py`

```python
"""Measure an on-axis PSF through a Fresnel lens."""
import bpy

from fresnel_imaging.simulation.fresnel_lens import create_fresnel_lens
from fresnel_imaging.simulation.materials import create_glass_material, assign_material
from fresnel_imaging.simulation.psf_measurement import setup_psf_scene
from fresnel_imaging.simulation.render import render_scene

# ── Clean scene ─────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# ── Build lens ──────────────────────────────────────────────────────
lens = create_fresnel_lens(
    diameter=100.0, focal_length=200.0, n_grooves=50, thickness=2.0,
    material_ior=1.4917, name="FresnelLens",
)
glass = create_glass_material(name="PMMA", ior=1.4917, cauchy_b=0.0)
assign_material(lens, glass)

# ── Assemble PSF scene ──────────────────────────────────────────────
psf_info = setup_psf_scene(
    lens_obj=lens,
    point_position=(0.0, 0.0, -500.0),   # on-axis
    sensor_distance=None,                  # auto from thin lens eq
    render_samples=5000,                   # high SPP for clean PSF
    render_resolution=(512, 512),          # square crop
)

print(f"Sensor distance: {psf_info['sensor_distance']:.2f} mm")
print(f"Engine: {psf_info['render_config']['engine']}")

# ── Render to EXR ───────────────────────────────────────────────────
render_scene("output/psf_on_axis.exr")
print("PSF render complete.")
```

```bash
blender -b -P measure_psf.py
```

`setup_psf_scene` creates:
- An **emissive UV sphere** (point source) via `create_point_source`
  — uses an emission shader rather than a Blender light so that the
  light correctly refracts through LuxCore glass geometry.
- A **camera** at the computed image plane.
- Render settings configured for EXR output with 32-bit colour depth.

### 6.2 Extracting the PSF Kernel

After rendering, extract a normalised PSF kernel.  This step is pure
NumPy — it does **not** require Blender.

```python
from fresnel_imaging.simulation.psf_measurement import extract_psf_from_render

# From an EXR file
psf = extract_psf_from_render(
    source="output/psf_on_axis.exr",
    psf_size=31,        # 31×31 pixel kernel
    threshold=0.001,    # minimum total intensity
)

print(f"PSF shape: {psf.shape}")    # (31, 31)
print(f"PSF sum:   {psf.sum():.6f}")  # ≈ 1.0
print(f"PSF dtype: {psf.dtype}")      # float64
```

`extract_psf_from_render` accepts:
- A **file path** (str or Path) — loaded via OpenCV with full bit depth.
- A **NumPy array** — used directly.

The function:
1. Converts to grayscale (luminance weighting).
2. Finds the intensity centroid.
3. Crops a `psf_size × psf_size` patch centred on the centroid.
4. Clips negatives and normalises to sum = 1.0.

If total intensity is below `threshold`, returns a uniform kernel
(avoids degenerate centroid computation on black renders).

### 6.3 Spatially-Varying PSF (Grid Measurement)

Real lenses have PSFs that vary across the field of view.  The function
`setup_psf_grid_scene` computes a grid of point-source positions.

```python
from fresnel_imaging.simulation.psf_measurement import (
    setup_psf_grid_scene,
    setup_psf_scene,
    extract_psf_from_render,
)
from fresnel_imaging.simulation.render import render_scene
import numpy as np

# ── Get grid positions ──────────────────────────────────────────────
grid_info = setup_psf_grid_scene(
    lens_obj=lens,
    grid_rows=5,
    grid_cols=5,
    grid_spacing=20.0,         # 20 mm between points
    object_distance=500.0,
    render_samples=5000,
    render_resolution=(512, 512),
)

positions = grid_info["grid_positions"]
print(f"Grid: {grid_info['grid_shape']} = {len(positions)} positions")
print(f"Sensor distance: {grid_info['sensor_distance']:.2f} mm")

# ── Render each point source one at a time ──────────────────────────
psf_list = []
for idx, pos in enumerate(positions):
    # Clean previous point sources
    for obj in bpy.data.objects:
        if obj.name.startswith("PointSource"):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Set up scene for this position
    psf_info = setup_psf_scene(
        lens_obj=lens,
        point_position=pos,
        sensor_distance=grid_info["sensor_distance"],
        render_samples=5000,
        render_resolution=(512, 512),
    )

    # Render
    out_path = f"output/psf_grid/psf_{idx:03d}.exr"
    render_scene(out_path)

    # Extract kernel
    psf = extract_psf_from_render(out_path, psf_size=31)
    psf_list.append(psf)
    print(f"  [{idx+1}/{len(positions)}] position={pos}, peak={psf.max():.4f}")

# Stack into array for downstream use
psf_grid = np.stack(psf_list)  # shape (25, 31, 31)
np.save("output/psf_grid.npy", psf_grid)
```

> **Important:** Render ONE point at a time to avoid PSF overlap.
> The `setup_psf_grid_scene` function returns an `"instructions"` key
> reminding you of this.

---

## 7. Deconvolution

### 7.1 Single-Channel: `fast_deconv`

The hyper-Laplacian deconvolution solver from Krishnan & Fergus (2009).

```python
import numpy as np
from PIL import Image
from fresnel_imaging.deconvolution.fast_deconv import fast_deconv

# Load blurred image (grayscale)
blurred = np.array(Image.open("blurred.jpg").convert("L")).astype(np.float64) / 255.0

# Load PSF kernel
psf = np.load("output/psf_grid.npy")[12]  # centre PSF from 5×5 grid
psf = psf / psf.sum()  # ensure normalised

# Deconvolve
restored = fast_deconv(
    yin=blurred,            # (M, N) grayscale
    k=psf,                  # (K, K) kernel — must be odd-sized
    lambda_param=2000.0,    # data fidelity weight
    alpha=2.0/3.0,          # hyper-Laplacian exponent (2/3 is optimal)
)

# Save result
restored_uint8 = np.clip(restored * 255, 0, 255).astype(np.uint8)
Image.fromarray(restored_uint8).save("restored_gray.png")
```

**Parameters:**
- `lambda_param` — higher = sharper but more ringing.  Start at 2000,
  tune by inspection.
- `alpha` — 2/3 is the statistically optimal exponent for natural
  images.  Use 1.0 for TV regularisation.
- `yout0` — optional initialisation (defaults to the input image).

### 7.2 Multi-Channel (Colour): `fast_deconv_yuv`

Handles per-channel blur kernels in YUV colour space.

```python
import numpy as np
from PIL import Image
from fresnel_imaging.deconvolution.fast_deconv_yuv import fast_deconv_yuv

# Load blurred RGB image
blurred = np.array(Image.open("blurred.jpg")).astype(np.float64) / 255.0

# Per-channel PSFs (may differ due to chromatic aberration)
psf_r = np.load("psf_red.npy")
psf_g = np.load("psf_green.npy")
psf_b = np.load("psf_blue.npy")

restored = fast_deconv_yuv(
    yin=blurred,                        # (M, N, 3) RGB image
    k=[psf_r, psf_g, psf_b],           # per-channel kernels
    lambda_param=2000.0,
    rho_yuv=[1.0, 1.0, 1.0],           # per-channel prior weights (YUV)
    w_rgb=[1.0, 1.0, 1.0],             # per-channel Tikhonov weights (RGB)
    theta=0.001,                        # Tikhonov regularisation strength
    alpha=0.65,                         # hyper-Laplacian exponent
)

restored_uint8 = np.clip(restored * 255, 0, 255).astype(np.uint8)
Image.fromarray(restored_uint8).save("restored_color.png")
```

**Notes:**
- The solver works in YUV space internally using the ITU-R BT.601
  `RGB_TO_YUV` matrix.
- Edge tapering is applied automatically (4 passes via `edgetaper`).
- Padding is added and stripped automatically.
- Each kernel must be odd-sized; a `ValueError` is raised otherwise.

### 7.3 Spatially-Varying Deconvolution

For lenses with strong field-dependent aberrations:

```python
import numpy as np
from functools import partial
from fresnel_imaging.deconvolution.fast_deconv import fast_deconv
from fresnel_imaging.calibration.pipeline import (
    apply_spatially_varying_deconv,
    load_calibration,
)

# Load calibration data
cal = load_calibration("calibration.npz")

# Define a deconvolution function (fixed lambda and alpha)
deconv_fn = partial(fast_deconv, lambda_param=2000.0, alpha=2.0/3.0)

# Apply spatially-varying deconvolution
blurred = np.array(Image.open("blurred.jpg").convert("L")).astype(np.float64) / 255.0

restored = apply_spatially_varying_deconv(
    image=blurred,
    psf_grid=cal["psf_grid"],
    positions=cal["positions"],
    deconv_fn=deconv_fn,
    patch_size=64,
    overlap=0.5,
)
```

`apply_spatially_varying_deconv` splits into overlapping patches,
uses local PSFs, deconvolves patch-by-patch, then blends seams.

---

## 8. Complete Pipeline Example

```python
"""End-to-end Fresnel imaging pipeline.

Run inside Blender:
    blender -b -P pipeline.py
"""
import bpy
import numpy as np

from fresnel_imaging.simulation.fresnel_lens import create_fresnel_lens
from fresnel_imaging.simulation.materials import create_glass_material, assign_material
from fresnel_imaging.simulation.scene_setup import setup_imaging_scene
from fresnel_imaging.simulation.psf_measurement import setup_psf_scene, extract_psf_from_render
from fresnel_imaging.simulation.render import render_scene, configure_luxcore_render
from fresnel_imaging.deconvolution.fast_deconv import fast_deconv

# STEP 1: Build the lens
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

lens = create_fresnel_lens(
    diameter=100.0, focal_length=200.0, n_grooves=50,
    thickness=2.0, material_ior=1.4917,
)
glass = create_glass_material(name="PMMA", ior=1.4917, cauchy_b=0.0)
assign_material(lens, glass)

# STEP 2: Render a blurred image of a target
scene_info = setup_imaging_scene(
    lens_obj=lens, image_path="target.jpg", distance=1000.0,
)
configure_luxcore_render(samples=2000, engine="BIDIR", resolution=(1024, 768))
render_scene("output/blurred.exr")

# STEP 3: Measure the PSF (clean scene, re-use same lens)
# Remove imaging scene objects but keep the lens
for obj in list(bpy.data.objects):
    if obj.name != lens.name:
        bpy.data.objects.remove(obj, do_unlink=True)

psf_info = setup_psf_scene(
    lens_obj=lens, point_position=(0.0, 0.0, -1000.0),
    render_samples=5000, render_resolution=(512, 512),
)
render_scene("output/psf.exr")

psf = extract_psf_from_render("output/psf.exr", psf_size=31)

# STEP 4: Deconvolve (pure NumPy — no Blender needed)
import cv2  # noqa: E402

blurred = cv2.imread("output/blurred.exr", cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float64)

restored = fast_deconv(yin=gray, k=psf, lambda_param=2000.0, alpha=2.0/3.0)

restored_uint8 = np.clip(restored * 255, 0, 255).astype(np.uint8)
cv2.imwrite("output/restored.png", restored_uint8)
print("Pipeline complete.  See output/ directory.")
```

---

## 9. Troubleshooting

### Black renders (no light reaching the sensor)

| Cause | Fix |
|-------|-----|
| Object distance ≤ focal length | Move the object farther from the lens (d₀ > f). |
| Camera not assigned to scene | Ensure `bpy.context.scene.camera = cam_obj`. |
| Emission strength too low | Increase `emission_strength` in `create_point_source` (default 10 000). |
| Bounce depth too low | Set `light_depth` and `path_depth` ≥ 15 in `configure_luxcore_render`. |
| Glass material not assigned | Verify `assign_material(lens, glass)` was called. |

### Noisy PSF

| Cause | Fix |
|-------|-----|
| Too few samples | Increase `render_samples` to 5 000–10 000 for PSF scenes. |
| PATH engine | Switch to `BIDIR` — it handles caustics through glass far better. |
| Small point source | Default radius 0.1 mm is fine; don't go below 0.05. |

### Mesh artefacts (holes, flipped normals)

| Cause | Fix |
|-------|-----|
| Semi-diameter ≥ R of curvature | Reduce `diameter` or increase `focal_length`. |
| Too few segments | Increase `segments` from 128 to 256. |
| Duplicate vertices | The `_revolve_profile` function calls `remove_doubles`; if issues persist, increase `dist` tolerance. |

### Import errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: This module must be run inside Blender` | Running simulation code outside Blender | Use `blender -b -P script.py` |
| `ModuleNotFoundError: No module named 'bpy'` | Same as above | Same as above |
| `ModuleNotFoundError: No module named 'fresnel_imaging'` | Package not installed in Blender's Python | Run `pip install -e .` using Blender's bundled Python, or set `PYTHONPATH` |

## Appendix A: Glass Catalog

| Glass Name | IOR (n_d @ 587.6 nm) | Cauchy-B | Typical Use |
|------------|----------------------|----------|-------------|
| **BK7** | 1.5168 | 0.00420 | General-purpose crown glass; doublet front elements |
| **SF5** | 1.6727 | 0.00896 | Dense flint glass; doublet rear elements (high dispersion) |
| **LAK9** | 1.6910 | 0.00531 | Lanthanum crown; high-index low-dispersion objectives |
| **FUSED_SILICA** | 1.4585 | 0.00354 | UV-grade optics; laser windows |
| **PMMA** | 1.4917 | 0.00000 | Acrylic / plexiglass; Fresnel lenses |

---

## Appendix B: LuxCore Settings Reference

All parameters for `configure_luxcore_render`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `samples` | int | 2000 | Halt sample count (samples per pixel). Higher = less noise, slower. |
| `engine` | str | `"BIDIR"` | `"BIDIR"` (bidirectional — best for caustics) or `"PATH"` (unidirectional). |
| `resolution` | tuple | `(1920, 1080)` | Output image resolution `(width, height)` in pixels. |
| `light_depth` | int | 15 | Maximum bounce depth for light sub-paths (BIDIR only). |
| `path_depth` | int | 15 | Maximum bounce depth for camera sub-paths. |
| `sampler` | str | `"METROPOLIS"` | `"METROPOLIS"` (MLT, best for caustics), `"SOBOL"`, or `"RANDOM"`. |
| `use_photongi` | bool | `False` | Enable PhotonGI cache (PATH engine only). |
| `photongi_count` | float | 20.0 | Photon count in millions when PhotonGI is enabled. |

---

## Appendix C: Source and License Notes

- Canonical repository: <https://github.com/simonseo/fresnel-imaging>
- Created with help from OpenCode.
- Original algorithm sources: Seo ("Computational Imaging with Fresnel Lenses"), Heide et al. (SIGGRAPH 2013), Krishnan & Fergus (NIPS 2009), Schuler et al. (ICCV 2011), Mannan & Langer (CRV 2016).
- Code in this repository is a Python implementation informed by those publications and prior MATLAB pipeline behavior.
- External dependencies (NumPy, SciPy, OpenCV, Blender, LuxCore) are distributed under their respective upstream licenses.
