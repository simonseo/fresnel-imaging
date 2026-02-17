from pathlib import Path
from typing import Any

bpy = __import__("bpy")


ROOT = Path("/Users/sseo/Documents/fresnel-imaging")
TRIPLET_DIR = ROOT / "project" / "assets" / "lenses" / "triplet_a"
STEP4_DIR = TRIPLET_DIR / "step4"
STEP4_DIR.mkdir(parents=True, exist_ok=True)


def _detect_luxcore(scene: Any) -> bool:
    try:
        scene.render.engine = "LUXCORE"
        return True
    except Exception:
        return False


def _set_common_render(scene: Any) -> None:
    scene.cycles.samples = 256
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000


def _lens_objects() -> list[Any]:
    return [obj for obj in bpy.data.objects if obj.name.startswith("triplet_a_E")]


bpy.ops.wm.open_mainfile(filepath=str(TRIPLET_DIR / "scene.blend"))
scene = bpy.context.scene
luxcore_available = _detect_luxcore(scene)
scene.render.engine = "CYCLES"
_set_common_render(scene)

# Sharp reference (no lens)
for lens in _lens_objects():
    lens.hide_render = True
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(STEP4_DIR / "reference_scene.png")
bpy.ops.render.render(write_still=True)

# Blurred render (lens enabled)
for lens in _lens_objects():
    lens.hide_render = False
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.filepath = str(STEP4_DIR / "blurred_scene.exr")
bpy.ops.render.render(write_still=True)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(STEP4_DIR / "blurred_scene.png")
bpy.ops.render.render(write_still=True)

# Point-source PSF captures: isolate lens + emitter only
for obj in bpy.data.objects:
    if obj.type == "LIGHT":
        obj.hide_render = True

for name in [
    "CornellFloor",
    "CornellCeiling",
    "CornellBackWall",
    "CornellLeftWall",
    "CornellRightWall",
]:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        obj.hide_render = True

scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg is not None:
    bg.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs[1].default_value = 0.0

bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0.0, 190.0, 27.0))
point_center = bpy.context.active_object
point_center.name = "PointSourceCenter"

mat = bpy.data.materials.new(name="PointEmitter")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()
out = nodes.new(type="ShaderNodeOutputMaterial")
emission = nodes.new(type="ShaderNodeEmission")
emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
emission.inputs["Strength"].default_value = 8000.0
links.new(emission.outputs["Emission"], out.inputs["Surface"])
point_center.data.materials.append(mat)

scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.filepath = str(STEP4_DIR / "psf_point_center.exr")
bpy.ops.render.render(write_still=True)

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(STEP4_DIR / "psf_point_center.png")
bpy.ops.render.render(write_still=True)

point_center.location.x = 12.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.filepath = str(STEP4_DIR / "psf_point_offaxis.exr")
bpy.ops.render.render(write_still=True)

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(STEP4_DIR / "psf_point_offaxis.png")
bpy.ops.render.render(write_still=True)

print(f"engine=CYCLES luxcore_available={luxcore_available}")
print(str(STEP4_DIR / "reference_scene.png"))
print(str(STEP4_DIR / "blurred_scene.exr"))
print(str(STEP4_DIR / "blurred_scene.png"))
print(str(STEP4_DIR / "psf_point_center.png"))
print(str(STEP4_DIR / "psf_point_offaxis.png"))
