import math
import sys

import bpy

sys.path.insert(0, "/Users/sseo/Documents/fresnel-imaging/src/fresnel_imaging/simulation")
import fresnel_lens

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")

lens = fresnel_lens.create_fresnel_lens(
    diameter=100.0,
    focal_length=200.0,
    n_grooves=70,
    thickness=2.2,
    name="PreviewFresnel",
    segments=256,
)

material = bpy.data.materials.new(name="LensPreviewMat")
material.use_nodes = True
nodes = material.node_tree.nodes
links = material.node_tree.links
nodes.clear()
out = nodes.new(type="ShaderNodeOutputMaterial")
bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.82, 1.0)
bsdf.inputs["Roughness"].default_value = 0.35
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
lens.data.materials.clear()
lens.data.materials.append(material)

bpy.ops.mesh.primitive_plane_add(size=260, location=(0.0, 0.0, -1.2))
plane = bpy.context.active_object
plane_mat = bpy.data.materials.new(name="Ground")
plane_mat.use_nodes = True
plane_nodes = plane_mat.node_tree.nodes
plane_links = plane_mat.node_tree.links
plane_nodes.clear()
plane_out = plane_nodes.new(type="ShaderNodeOutputMaterial")
plane_bsdf = plane_nodes.new(type="ShaderNodeBsdfPrincipled")
plane_bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
plane_bsdf.inputs["Roughness"].default_value = 0.9
plane_links.new(plane_bsdf.outputs["BSDF"], plane_out.inputs["Surface"])
plane.data.materials.append(plane_mat)

camera_data = bpy.data.cameras.new("Cam")
camera = bpy.data.objects.new("Cam", camera_data)
bpy.context.collection.objects.link(camera)
camera.location = (120.0, -180.0, 70.0)
track = camera.constraints.new(type="TRACK_TO")
track.target = lens
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"
scene.camera = camera

sun_data = bpy.data.lights.new(name="Sun", type="SUN")
sun_data.energy = 4.0
sun = bpy.data.objects.new(name="Sun", object_data=sun_data)
bpy.context.collection.objects.link(sun)
sun.location = (150.0, -150.0, 220.0)
sun.rotation_euler = (math.radians(45), 0.0, math.radians(35))

area_data = bpy.data.lights.new(name="Fill", type="AREA")
area_data.energy = 1200.0
area_data.size = 90.0
area = bpy.data.objects.new(name="Fill", object_data=area_data)
bpy.context.collection.objects.link(area)
area.location = (-140.0, 130.0, 100.0)
area.rotation_euler = (math.radians(70), 0.0, math.radians(-130))

world = scene.world
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
if background is not None:
    background.inputs[0].default_value = (0.66, 0.66, 0.66, 1.0)
    background.inputs[1].default_value = 1.0

scene.render.engine = "CYCLES"
scene.cycles.samples = 128
scene.render.resolution_x = 1600
scene.render.resolution_y = 1000
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = "/Users/sseo/Documents/fresnel-imaging/outputs/fresnel_preview_geometry.png"
bpy.ops.render.render(write_still=True)

print("Saved /Users/sseo/Documents/fresnel-imaging/outputs/fresnel_preview_geometry.png")
