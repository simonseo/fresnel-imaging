import json
from pathlib import Path

import cv2
import numpy as np

from fresnel_imaging.simulation.psf_measurement import extract_psf_from_render

ROOT = Path("/Users/sseo/Documents/fresnel-imaging")
STEP4_DIR = ROOT / "project" / "assets" / "lenses" / "triplet_a" / "step4"

center = extract_psf_from_render(STEP4_DIR / "psf_point_center.png", psf_size=31)
offaxis = extract_psf_from_render(STEP4_DIR / "psf_point_offaxis.png", psf_size=31)

np.save(STEP4_DIR / "psf_center.npy", center)
np.save(STEP4_DIR / "psf_offaxis.npy", offaxis)

center_png = np.clip(center / max(center.max(), 1e-12) * 255.0, 0, 255).astype(np.uint8)
offaxis_png = np.clip(offaxis / max(offaxis.max(), 1e-12) * 255.0, 0, 255).astype(np.uint8)

cv2.imwrite(str(STEP4_DIR / "psf_center_preview.png"), center_png)
cv2.imwrite(str(STEP4_DIR / "psf_offaxis_preview.png"), offaxis_png)

payload = {
    "psf_size": 31,
    "center_sum": float(center.sum()),
    "offaxis_sum": float(offaxis.sum()),
    "center_max": float(center.max()),
    "offaxis_max": float(offaxis.max()),
    "inputs": {
        "center_png": "psf_point_center.png",
        "offaxis_png": "psf_point_offaxis.png",
    },
    "outputs": {
        "center_npy": "psf_center.npy",
        "offaxis_npy": "psf_offaxis.npy",
        "center_preview_png": "psf_center_preview.png",
        "offaxis_preview_png": "psf_offaxis_preview.png",
    },
}

(STEP4_DIR / "psf_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
