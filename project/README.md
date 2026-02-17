# Project Workspace

This folder is a build-on-top workspace for the `fresnel_imaging` library.

- `code/` reproducible asset-generation scripts.
- `assets/` generated models, previews, metadata, and Blender scenes.
- `references/` source notes for papers and external resources.

Generate assets:

```bash
"/Applications/Blender.app/Contents/MacOS/Blender" -b -P project/code/generate_lens_assets.py
```

This script generates:

- A modular Cornell box asset under `project/assets/common/`.
- Six lens asset sets under `project/assets/lenses/`:
  - `fresnel_a`, `fresnel_b`, `fresnel_c`
  - `biconvex_a`, `biconvex_b`
  - `triplet_a`

For full steps 1-8 execution (toolchain check, step-4 PSFs, step-5/6
calibration+deconvolution, and validation), see
`project/WORKFLOW_PROJECT.md`.
