import json
import math
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

bpy = __import__("bpy")

ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = ROOT / "project" / "assets"
COMMON_ROOT = ASSETS_ROOT / "common"
LENSES_ROOT = ASSETS_ROOT / "lenses"
SIM_ROOT = ROOT / "src" / "fresnel_imaging" / "simulation"


def load_simulation_modules() -> tuple[types.ModuleType, types.ModuleType]:
    if "fresnel_imaging" not in sys.modules:
        pkg = types.ModuleType("fresnel_imaging")
        pkg.__path__ = [str(ROOT / "src" / "fresnel_imaging")]
        sys.modules["fresnel_imaging"] = pkg
    if "fresnel_imaging.simulation" not in sys.modules:
        sim_pkg = types.ModuleType("fresnel_imaging.simulation")
        sim_pkg.__path__ = [str(SIM_ROOT)]
        sys.modules["fresnel_imaging.simulation"] = sim_pkg

    fresnel_module = _load_module(
        "fresnel_imaging.simulation.fresnel_lens",
        SIM_ROOT / "fresnel_lens.py",
    )
    _load_module(
        "fresnel_imaging.simulation.materials",
        SIM_ROOT / "materials.py",
    )
    lens_module = _load_module(
        "fresnel_imaging.simulation.lens_geometry",
        SIM_ROOT / "lens_geometry.py",
    )
    return fresnel_module, lens_module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {name}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reset_scene() -> Any:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    return scene


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return material


def add_plane(
    name: str,
    location: tuple[float, float, float],
    rotation: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: Any,
) -> Any:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_cornell_box() -> dict[str, Any]:
    red = make_material("CornellRed", (0.63, 0.08, 0.08, 1.0), 0.95)
    green = make_material("CornellGreen", (0.1, 0.52, 0.14, 1.0), 0.95)
    white = make_material("CornellWhite", (0.74, 0.74, 0.74, 1.0), 0.95)

    w = 120.0
    d = 150.0
    h = 95.0

    floor = add_plane("CornellFloor", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (w, d, 1.0), white)
    ceiling = add_plane(
        "CornellCeiling",
        (0.0, 0.0, h),
        (math.radians(180.0), 0.0, 0.0),
        (w, d, 1.0),
        white,
    )
    back = add_plane(
        "CornellBackWall",
        (0.0, d, h / 2.0),
        (math.radians(90.0), 0.0, 0.0),
        (w, h / 2.0, 1.0),
        white,
    )
    left = add_plane(
        "CornellLeftWall",
        (-w, 0.0, h / 2.0),
        (0.0, math.radians(90.0), math.radians(90.0)),
        (d, h / 2.0, 1.0),
        red,
    )
    right = add_plane(
        "CornellRightWall",
        (w, 0.0, h / 2.0),
        (0.0, math.radians(90.0), math.radians(-90.0)),
        (d, h / 2.0, 1.0),
        green,
    )

    light_data = bpy.data.lights.new(name="CornellLight", type="AREA")
    light_data.energy = 1500.0
    light_data.shape = "RECTANGLE"
    light_data.size = 45.0
    light_data.size_y = 30.0
    light = bpy.data.objects.new(name="CornellLight", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, 30.0, h - 2.0)
    light.rotation_euler = (math.radians(180.0), 0.0, 0.0)

    return {
        "floor": floor,
        "ceiling": ceiling,
        "back": back,
        "left": left,
        "right": right,
        "light": light,
    }


def setup_camera(target: Any) -> Any:
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, -240.0, 55.0)
    track = camera.constraints.new(type="TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.camera = camera
    return camera


def apply_world_background() -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.52, 0.52, 0.52, 1.0)
        bg.inputs[1].default_value = 1.0


def apply_lens_preview_material(objs: list[Any]) -> None:
    mat = make_material("LensPreview", (0.78, 0.78, 0.78, 1.0), 0.3)
    for obj in objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def make_glass_material() -> Any:
    material = bpy.data.materials.new(name="LensGlass")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = 1.49
    glass.inputs["Roughness"].default_value = 0.0
    links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    return material


def apply_glass_material(objs: list[Any]) -> None:
    mat = make_glass_material()
    for obj in objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def set_render_settings(samples: int = 96) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1000
    scene.render.image_settings.file_format = "PNG"


def select_only(objs: list[Any]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def export_lens_models(objs: list[Any], out_dir: Path, stem: str) -> None:
    select_only(objs)
    bpy.ops.export_scene.gltf(
        filepath=str(out_dir / f"{stem}.glb"),
        export_format="GLB",
        use_selection=True,
    )
    bpy.ops.export_scene.gltf(
        filepath=str(out_dir / f"{stem}.gltf"),
        export_format="GLTF_SEPARATE",
        use_selection=True,
    )


def render_preview(out_path: Path) -> None:
    scene = bpy.context.scene
    scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def place_lens_objects(objs: list[Any]) -> None:
    for obj in objs:
        obj.location = (0.0, 45.0, 27.0)
        obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)


def generate_cornell_box_assets() -> None:
    COMMON_ROOT.mkdir(parents=True, exist_ok=True)
    reset_scene()
    cornell = create_cornell_box()
    setup_camera(cornell["back"])
    apply_world_background()
    set_render_settings(samples=48)
    render_preview(COMMON_ROOT / "cornell_box_preview.png")
    select_only(
        [
            cornell["floor"],
            cornell["ceiling"],
            cornell["back"],
            cornell["left"],
            cornell["right"],
        ]
    )
    bpy.ops.export_scene.gltf(
        filepath=str(COMMON_ROOT / "cornell_box.glb"),
        export_format="GLB",
        use_selection=True,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(COMMON_ROOT / "cornell_box.blend"))


def lens_specs() -> list[dict[str, object]]:
    return [
        {
            "id": "fresnel_a",
            "kind": "fresnel",
            "params": {
                "diameter": 100.0,
                "focal_length": 200.0,
                "n_grooves": 60,
                "thickness": 2.2,
                "segments": 256,
            },
        },
        {
            "id": "fresnel_b",
            "kind": "fresnel",
            "params": {
                "diameter": 80.0,
                "focal_length": 130.0,
                "n_grooves": 80,
                "thickness": 1.8,
                "segments": 256,
            },
        },
        {
            "id": "fresnel_c",
            "kind": "fresnel",
            "params": {
                "diameter": 120.0,
                "focal_length": 300.0,
                "n_grooves": 72,
                "thickness": 2.8,
                "segments": 256,
            },
        },
        {
            "id": "biconvex_a",
            "kind": "biconvex",
            "params": {
                "prescription": [
                    {"radius": 70.0, "thickness": 7.0, "ior": 1.5168, "diameter": 52.0},
                    {"radius": -70.0, "thickness": 25.0, "ior": 1.0, "diameter": 52.0},
                ],
                "segments": 256,
            },
        },
        {
            "id": "biconvex_b",
            "kind": "biconvex",
            "params": {
                "prescription": [
                    {"radius": 45.0, "thickness": 5.5, "ior": 1.62, "diameter": 40.0},
                    {"radius": -52.0, "thickness": 22.0, "ior": 1.0, "diameter": 40.0},
                ],
                "segments": 256,
            },
        },
        {
            "id": "triplet_a",
            "kind": "triplet",
            "params": {
                "prescription": [
                    {"radius": 80.0, "thickness": 4.0, "ior": 1.62, "diameter": 48.0},
                    {"radius": -65.0, "thickness": 2.0, "ior": 1.0, "diameter": 48.0},
                    {"radius": 42.0, "thickness": 2.4, "ior": 1.72, "diameter": 40.0},
                    {"radius": -38.0, "thickness": 1.5, "ior": 1.0, "diameter": 40.0},
                    {"radius": 65.0, "thickness": 3.8, "ior": 1.62, "diameter": 44.0},
                    {"radius": -90.0, "thickness": 25.0, "ior": 1.0, "diameter": 44.0},
                ],
                "segments": 256,
            },
        },
    ]


def build_lens(
    spec: dict[str, Any],
    fresnel_module: types.ModuleType,
    lens_module: types.ModuleType,
) -> list[Any]:
    lens_id = spec["id"]
    kind = spec["kind"]
    params = spec["params"]
    if kind == "fresnel":
        obj = fresnel_module.create_fresnel_lens(name=lens_id, **params)
        return [obj]
    objs = lens_module.create_lens_from_prescription(
        prescription=params["prescription"],
        name=lens_id,
        segments=params["segments"],
    )
    return list(objs)


def write_metadata(
    spec: dict[str, Any],
    lens_objects: list[Any],
    out_dir: Path,
) -> None:
    payload = {
        "id": spec["id"],
        "kind": spec["kind"],
        "parameters": spec["params"],
        "object_names": [obj.name for obj in lens_objects],
        "object_count": len(lens_objects),
        "files": {
            "scene_blend": "scene.blend",
            "preview_png": "preview.png",
            "model_glb": "lens.glb",
            "model_gltf": "lens.gltf",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_lens_assets() -> None:
    fresnel_module, lens_module = load_simulation_modules()
    LENSES_ROOT.mkdir(parents=True, exist_ok=True)

    for spec in lens_specs():
        out_dir = LENSES_ROOT / str(spec["id"])
        out_dir.mkdir(parents=True, exist_ok=True)

        scene = reset_scene()
        create_cornell_box()
        apply_world_background()
        lens_objects = build_lens(spec, fresnel_module, lens_module)
        place_lens_objects(lens_objects)
        apply_lens_preview_material(lens_objects)
        setup_camera(lens_objects[0])
        set_render_settings(samples=96)
        render_preview(out_dir / "preview.png")

        apply_glass_material(lens_objects)
        export_lens_models(lens_objects, out_dir, "lens")

        scene.render.filepath = str(out_dir / "scene_preview.png")
        bpy.ops.render.render(write_still=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))
        write_metadata(spec, lens_objects, out_dir)


def main() -> None:
    generate_cornell_box_assets()
    generate_lens_assets()
    print(f"Generated assets in {ASSETS_ROOT}")


if __name__ == "__main__":
    main()
