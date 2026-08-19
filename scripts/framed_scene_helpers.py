#!/usr/bin/env python3
"""Shared Blender 5.2 Eevee scene-construction helpers.

This module intentionally does not save, render, or pack files. The project
entry point is ``build_project.py``, which creates separate, portable Blender
files that all reference the sibling ``assets/`` directory.
"""

from __future__ import annotations

from math import atan2
from typing import Iterable

import bpy
from mathutils import Vector


IMAGE_ASPECT = 3.0 / 4.0
CARD_HEIGHT = 1.20
CARD_WIDTH = CARD_HEIGHT * IMAGE_ASPECT
FRAME_BORDER = 0.075
FRAME_DEPTH = 0.055
CARD_DEPTH = 0.012
CARD_FRONT_GAP = 0.003
WALL_DEPTH = 0.035
WALL_Z = -0.025
HDRI_STRENGTH = 0.70
RENDER_SAMPLES = 64
RENDER_SIZE = 2048


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_box(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int, int]],
    face_materials: list[int],
    face_uvs: list[list[tuple[float, float]]],
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material_index: int,
    front_uv: bool = False,
) -> None:
    """Append a cuboid; the +Z face gets the portrait UVs when requested."""
    cx, cy, cz = center
    dx, dy, dz = (dimension / 2.0 for dimension in dimensions)
    start = len(vertices)
    vertices.extend(
        [
            (cx - dx, cy - dy, cz - dz),
            (cx + dx, cy - dy, cz - dz),
            (cx + dx, cy + dy, cz - dz),
            (cx - dx, cy + dy, cz - dz),
            (cx - dx, cy - dy, cz + dz),
            (cx + dx, cy - dy, cz + dz),
            (cx + dx, cy + dy, cz + dz),
            (cx - dx, cy + dy, cz + dz),
        ]
    )

    local_faces = [
        (0, 3, 2, 1),  # -Z
        (4, 5, 6, 7),  # +Z / camera-facing
        (0, 1, 5, 4),  # -Y
        (1, 2, 6, 5),  # +X
        (2, 3, 7, 6),  # +Y
        (3, 0, 4, 7),  # -X
    ]
    neutral_uv = [(0.0, 0.0)] * 4
    portrait_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for face_index, local_face in enumerate(local_faces):
        faces.append(tuple(start + index for index in local_face))
        face_materials.append(material_index)
        face_uvs.append(portrait_uv if front_uv and face_index == 1 else neutral_uv)


def make_framed_portrait_mesh() -> bpy.types.Object:
    outer_width = CARD_WIDTH + 2.0 * FRAME_BORDER
    outer_height = CARD_HEIGHT + 2.0 * FRAME_BORDER
    inner_width = CARD_WIDTH
    inner_height = CARD_HEIGHT

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    face_materials: list[int] = []
    face_uvs: list[list[tuple[float, float]]] = []

    # One mesh intentionally contains the card and all four walnut rails.
    add_box(
        vertices,
        faces,
        face_materials,
        face_uvs,
        (0.0, 0.0, FRAME_DEPTH - CARD_FRONT_GAP - CARD_DEPTH / 2.0),
        (CARD_WIDTH, CARD_HEIGHT, CARD_DEPTH),
        1,
        front_uv=True,
    )
    add_box(
        vertices,
        faces,
        face_materials,
        face_uvs,
        (-outer_width / 2.0 + FRAME_BORDER / 2.0, 0.0, FRAME_DEPTH / 2.0),
        (FRAME_BORDER, outer_height, FRAME_DEPTH),
        0,
    )
    add_box(
        vertices,
        faces,
        face_materials,
        face_uvs,
        (outer_width / 2.0 - FRAME_BORDER / 2.0, 0.0, FRAME_DEPTH / 2.0),
        (FRAME_BORDER, outer_height, FRAME_DEPTH),
        0,
    )
    add_box(
        vertices,
        faces,
        face_materials,
        face_uvs,
        (0.0, outer_height / 2.0 - FRAME_BORDER / 2.0, FRAME_DEPTH / 2.0),
        (inner_width, FRAME_BORDER, FRAME_DEPTH),
        0,
    )
    add_box(
        vertices,
        faces,
        face_materials,
        face_uvs,
        (0.0, -outer_height / 2.0 + FRAME_BORDER / 2.0, FRAME_DEPTH / 2.0),
        (inner_width, FRAME_BORDER, FRAME_DEPTH),
        0,
    )

    mesh = bpy.data.meshes.new("FramedPortraitMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(bpy.data.materials["DarkWalnutFrame"])
    mesh.materials.append(bpy.data.materials["SelfiePortrait"])

    uv_layer = mesh.uv_layers.new(name="PortraitUV")
    for polygon, material_index, polygon_uvs in zip(mesh.polygons, face_materials, face_uvs):
        polygon.material_index = material_index
        for loop_index, uv in zip(polygon.loop_indices, polygon_uvs):
            uv_layer.data[loop_index].uv = uv

    for polygon in mesh.polygons:
        polygon.use_smooth = False

    portrait = bpy.data.objects.new("FramedPortrait", mesh)
    bpy.context.scene.collection.objects.link(portrait)

    bevel = portrait.modifiers.new(name="SoftFrameEdges", type="BEVEL")
    bevel.width = 0.006
    bevel.segments = 3
    bevel.limit_method = "ANGLE"
    if hasattr(bevel, "harden_normals"):
        bevel.harden_normals = True
    return portrait


def make_portrait_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.new("SelfiePortrait")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (320, 0)
    shader.inputs["Metallic"].default_value = 0.0

    image_node = nodes.new("ShaderNodeTexImage")
    image_node.location = (-620, 100)
    image_node.image = image
    image_node.interpolation = "Linear"
    image_node.extension = "CLIP"

    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-620, -180)
    noise.inputs["Scale"].default_value = 260.0
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.65

    bump = nodes.new("ShaderNodeBump")
    bump.location = (-10, -180)
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.0006

    roughness_ramp = nodes.new("ShaderNodeValToRGB")
    roughness_ramp.location = (-300, -390)
    roughness_ramp.color_ramp.elements[0].position = 0.25
    roughness_ramp.color_ramp.elements[0].color = (0.47, 0.47, 0.47, 1.0)
    roughness_ramp.color_ramp.elements[1].position = 0.75
    roughness_ramp.color_ramp.elements[1].color = (0.58, 0.58, 0.58, 1.0)

    links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    links.new(noise.outputs["Fac"], roughness_ramp.inputs["Fac"])
    links.new(roughness_ramp.outputs["Color"], shader.inputs["Roughness"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_walnut_material() -> bpy.types.Material:
    material = bpy.data.materials.new("DarkWalnutFrame")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    assert shader is not None
    shader.inputs["Base Color"].default_value = (0.025, 0.007, 0.003, 1.0)
    shader.inputs["Roughness"].default_value = 0.48
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Specular IOR Level"].default_value = 0.30
    return material


def make_plaster_material() -> bpy.types.Material:
    material = bpy.data.materials.new("WarmPlaster")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    assert shader is not None
    shader.inputs["Roughness"].default_value = 0.82

    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "PlasterGrain"
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.62

    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (0.40, 0.31, 0.25, 1.0)
    color_ramp.color_ramp.elements[1].color = (0.64, 0.53, 0.44, 1.0)
    color_ramp.color_ramp.elements[0].position = 0.28
    color_ramp.color_ramp.elements[1].position = 0.72

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.014

    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def make_wall() -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    face_materials: list[int] = []
    face_uvs: list[list[tuple[float, float]]] = []
    add_box(
        vertices,
        faces,
        face_materials,
        face_uvs,
        (0.0, 0.0, WALL_Z),
        (5.0, 4.0, WALL_DEPTH),
        0,
    )
    mesh = bpy.data.meshes.new("WarmPlasterWallMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(bpy.data.materials["WarmPlaster"])
    wall = bpy.data.objects.new("WarmPlasterWall", mesh)
    bpy.context.scene.collection.objects.link(wall)
    return wall


def make_camera() -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("SharedCameraData")
    camera_data.lens = 70.0
    camera_data.sensor_width = 36.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.dof.use_dof = True
    camera_data.dof.aperture_fstop = 5.6
    camera = bpy.data.objects.new("SharedCamera", camera_data)
    camera.location = (0.18, 0.0, 3.30)
    target = Vector((0.0, 0.0, FRAME_DEPTH - CARD_FRONT_GAP))
    # The portrait lies in the XY plane. Set the camera's world-Y axis as the
    # image up direction explicitly; the usual track-quaternion helper assumes
    # world Z is up and introduces a 90-degree roll for this near-overhead shot.
    camera.rotation_euler = (
        0.0,
        atan2(camera.location.x - target.x, camera.location.z - target.z),
        0.0,
    )
    camera.data.dof.focus_distance = (camera.location - target).length
    bpy.context.scene.collection.objects.link(camera)
    return camera


def make_area_light() -> bpy.types.Object:
    light_data = bpy.data.lights.new("SoftKeyAreaData", type="AREA")
    light_data.energy = 90.0
    light_data.shape = "DISK"
    light_data.size = 1.5
    light_data.color = (1.0, 0.92, 0.84)
    light = bpy.data.objects.new("SoftKeyArea", light_data)
    light.location = (-1.35, 1.25, 2.35)
    look_at(light, Vector((0.0, 0.0, 0.0)))
    bpy.context.scene.collection.objects.link(light)
    return light


def make_world(hdri: bpy.types.Image) -> bpy.types.World:
    world = bpy.data.worlds.new("PolyHavenHDRIWorld")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.name = "StudioCountryHallHDRI"
    environment.image = hdri
    environment.location = (-420, 0)
    background = nodes.new("ShaderNodeBackground")
    background.location = (-80, 0)
    background.inputs["Strength"].default_value = HDRI_STRENGTH
    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (220, 0)
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return world


def configure_scene(scene: bpy.types.Scene, camera: bpy.types.Object, world: bpy.types.World) -> None:
    scene.camera = camera
    scene.world = world
    engine_items = scene.render.bl_rna.properties["engine"].enum_items
    available_engines = {item.identifier for item in engine_items}
    # Blender 5.2 exposes the Eevee Next engine under the legacy-compatible
    # BLENDER_EEVEE identifier; newer builds may expose the explicit name.
    if "BLENDER_EEVEE_NEXT" in available_engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in available_engines:
        scene.render.engine = "BLENDER_EEVEE"
    else:
        raise RuntimeError(f"No Eevee engine is available: {sorted(available_engines)}")
    scene.render.resolution_x = RENDER_SIZE
    scene.render.resolution_y = RENDER_SIZE
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "AgX"
    # Blender 5.2's bundled OCIO config exposes AgX with a single neutral
    # look. Keeping it at None makes the comparison deterministic across
    # Blender 5.2 installations rather than relying on a version-specific
    # contrast-look enum.
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.filepath = ""

    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        for attribute in ("taa_render_samples", "render_samples"):
            if hasattr(eevee, attribute):
                setattr(eevee, attribute, RENDER_SAMPLES)
                break
    scene["render_samples"] = RENDER_SAMPLES
    scene["pipeline_engine"] = "Eevee Next"
    scene["source_aspect"] = "3:4"


def create_scene(
    name: str,
    objects: Iterable[bpy.types.Object],
    camera: bpy.types.Object,
    world: bpy.types.World,
) -> bpy.types.Scene:
    scene = bpy.data.scenes.new(name)
    configure_scene(scene, camera, world)
    for obj in objects:
        scene.collection.objects.link(obj)
    return scene


def scene_object_summary(scene: bpy.types.Scene) -> dict[str, object]:
    objects = list(scene.collection.all_objects)
    by_type: dict[str, int] = {}
    for obj in objects:
        by_type[obj.type] = by_type.get(obj.type, 0) + 1
    return {
        "count": len(objects),
        "names": sorted(obj.name for obj in objects),
        "types": by_type,
    }


def assert_scene_contracts(soft_scene: bpy.types.Scene, hdri_scene: bpy.types.Scene) -> None:
    soft_objects = list(soft_scene.collection.all_objects)
    hdri_objects = list(hdri_scene.collection.all_objects)

    assert len(hdri_objects) == 2, scene_object_summary(hdri_scene)
    assert sorted(obj.type for obj in hdri_objects) == ["CAMERA", "MESH"]
    assert {obj.name for obj in hdri_objects} == {"SharedCamera", "FramedPortrait"}
    assert not any(obj.type in {"LIGHT", "EMPTY"} for obj in hdri_objects)

    assert len(soft_objects) == 4, scene_object_summary(soft_scene)
    assert sorted(obj.type for obj in soft_objects) == ["CAMERA", "LIGHT", "MESH", "MESH"]
    assert {obj.name for obj in soft_objects} == {
        "SharedCamera",
        "FramedPortrait",
        "WarmPlasterWall",
        "SoftKeyArea",
    }
    assert hdri_scene.camera is soft_scene.camera
    assert hdri_scene.world is soft_scene.world
    assert hdri_scene.world.node_tree.nodes.get("StudioCountryHallHDRI") is not None
