#!/usr/bin/env python3
"""Build four unpacked framed-selfie Blender 5.2 variants.

Build only (the default, no final renders)::

    blender -b --python artifacts/framed-selfie-eevee/scripts/build_project.py

Opt-in final renders, optionally scoped to one version::

    blender -b --python artifacts/framed-selfie-eevee/scripts/build_project.py -- --render
    blender -b --python artifacts/framed-selfie-eevee/scripts/build_project.py -- --scope 04_instant_print_cycles --render

Each invocation starts from a factory scene and writes one independent blend
file under ``blends/``. Every image reference is deliberately ``//../assets``
relative to the blend's directory, and no image is packed into a blend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSET_DIR = PROJECT_DIR / "assets"
BLENDS_DIR = PROJECT_DIR / "blends"
RENDERS_DIR = PROJECT_DIR / "renders"
MANIFESTS_DIR = PROJECT_DIR / "manifests"
SELFIE_PATH = ASSET_DIR / "selfie.png"
HDRI_PATH = ASSET_DIR / "studio_country_hall_4k.exr"
REL_SELFIE = "//../assets/selfie.png"
REL_HDRI = "//../assets/studio_country_hall_4k.exr"
SIZE = 2048
SAMPLES = 64
CYCLES_SAMPLES = 128
EXPECTED_SELFIE_SHA256 = "17bfef209000a6966063d5d319896b26d8585bd51b5026d7b76c023b8b2e6e90"
EXPECTED_HDRI_SHA256 = "2fc0b1c5fb241367c7f0615b99b17e3f802ca6e8da1d7796350450d574b1f18f"
HDRI_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/studio_country_hall_4k.exr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render the selected 2048px still(s).")
    parser.add_argument(
        "--scope",
        choices=(
            "all",
            "01_soft_staged",
            "02_hdri_only",
            "03_instant_print",
            "04_instant_print_cycles",
        ),
        default="all",
        help="Build only one variant, or all existing variants (default: all).",
    )
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs() -> None:
    expected_hashes = {
        SELFIE_PATH: EXPECTED_SELFIE_SHA256,
        HDRI_PATH: EXPECTED_HDRI_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing shared asset: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Unexpected SHA-256 for {path}: {actual_hash}")
    if bpy.app.version < (5, 2, 0):
        raise RuntimeError(f"Blender 5.2+ is required; found {bpy.app.version_string}")


def reset_factory() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_shared_assets() -> tuple[bpy.types.Image, bpy.types.Image]:
    selfie = bpy.data.images.load(str(SELFIE_PATH), check_existing=False)
    selfie.name = "SharedSelfie_3x4"
    selfie.filepath = REL_SELFIE
    hdri = bpy.data.images.load(str(HDRI_PATH), check_existing=False)
    hdri.name = "SharedStudioCountryHallHDRI"
    hdri.filepath = REL_HDRI
    try:
        hdri.colorspace_settings.name = "Linear Rec.709"
    except (AttributeError, TypeError):
        pass
    return selfie, hdri


def import_baseline():
    """Reuse the validated first-two scene construction without executing it."""
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        return importlib.import_module("framed_scene_helpers")
    finally:
        sys.path.pop(0)


def unlink_from_temporary_scene(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)


def link_scene(name: str, objects: list[bpy.types.Object], camera: bpy.types.Object, world: bpy.types.World):
    baseline = import_baseline()
    scene = baseline.create_scene(name, objects, camera, world)
    # The baseline helper uses the validated Eevee/camera/color-management controls.
    scene.render.resolution_x = SIZE
    scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = False
    scene["render_samples"] = SAMPLES
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        for prop in ("taa_render_samples", "render_samples"):
            if hasattr(eevee, prop):
                setattr(eevee, prop, SAMPLES)
                break
    return scene


def scene_objects(scene: bpy.types.Scene) -> list[bpy.types.Object]:
    return list(scene.collection.all_objects)


def object_summary(scene: bpy.types.Scene) -> dict[str, object]:
    objects = scene_objects(scene)
    types: dict[str, int] = {}
    for obj in objects:
        types[obj.type] = types.get(obj.type, 0) + 1
    return {"count": len(objects), "names": sorted(obj.name for obj in objects), "types": types}


def remove_other_scenes(keep: bpy.types.Scene) -> None:
    for scene in list(bpy.data.scenes):
        if scene is not keep:
            bpy.data.scenes.remove(scene)


def assert_unpacked(scene: bpy.types.Scene, expected_names: set[str]) -> None:
    objects = scene_objects(scene)
    assert {obj.name for obj in objects} == expected_names, object_summary(scene)
    assert all(image.packed_file is None for image in bpy.data.images)
    assert bpy.data.images["SharedSelfie_3x4"].filepath == REL_SELFIE
    assert bpy.data.images["SharedStudioCountryHallHDRI"].filepath == REL_HDRI
    assert scene.render.resolution_x == SIZE and scene.render.resolution_y == SIZE
    assert scene.render.film_transparent is False


def save_manifest(
    variant: str,
    scene: bpy.types.Scene,
    blend_path: Path,
    render_path: Path,
    build_seconds: float,
    render_seconds: float | None,
    notes: list[str],
) -> None:
    world_background = scene.world.node_tree.nodes.get("Background") if scene.world and scene.world.use_nodes else None
    materials = sorted({slot.material.name for obj in scene_objects(scene) for slot in obj.material_slots if slot.material})
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant": variant,
        "blender_version": bpy.app.version_string,
        "blend_path": blend_path.relative_to(PROJECT_DIR).as_posix(),
        "render_path": render_path.relative_to(PROJECT_DIR).as_posix(),
        "blend_sha256": sha256_file(blend_path),
        "render_sha256": sha256_file(render_path) if render_path.is_file() else None,
        "assets": {
            "selfie": {"path": REL_SELFIE, "sha256": EXPECTED_SELFIE_SHA256, "aspect": "3:4"},
            "hdri": {"path": REL_HDRI, "sha256": EXPECTED_HDRI_SHA256, "url": HDRI_URL},
            "packed": False,
        },
        "settings": {
            "engine": scene.render.engine,
            "engine_family": "Cycles" if scene.render.engine == "CYCLES" else "Eevee Next",
            "resolution": [SIZE, SIZE],
            "render_samples": scene.cycles.samples if scene.render.engine == "CYCLES" else SAMPLES,
            "source_aspect": "3:4",
            "image_format": "PNG RGB 16-bit",
            "camera": {
                "name": scene.camera.name,
                "lens_mm": scene.camera.data.lens,
                "f_stop": scene.camera.data.dof.aperture_fstop,
                "location": [round(float(v), 6) for v in scene.camera.location],
                "rotation_euler": [round(float(v), 6) for v in scene.camera.rotation_euler],
                "focus_distance": round(float(scene.camera.data.dof.focus_distance), 6),
            },
            "color_management": {
                "view_transform": scene.view_settings.view_transform,
                "look": scene.view_settings.look,
                "exposure": scene.view_settings.exposure,
                "gamma": scene.view_settings.gamma,
            },
            "world": {
                "hdri_rotation_degrees": 0.0,
                "strength": None if world_background is None else world_background.inputs["Strength"].default_value,
                "visible": True,
            },
            "materials": materials,
            "cycles": None if scene.render.engine != "CYCLES" else {
                "device": scene.cycles.device,
                "compute_backend": scene.get("cycles_compute_backend"),
                "device_name": scene.get("cycles_device_name"),
                "adaptive_sampling": scene.cycles.use_adaptive_sampling,
                "adaptive_threshold": scene.cycles.adaptive_threshold,
                "adaptive_min_samples": scene.cycles.adaptive_min_samples,
                "denoising": scene.cycles.use_denoising,
                "denoiser": scene.cycles.denoiser,
                "denoising_quality": scene.cycles.denoising_quality,
                "light_tree": scene.cycles.use_light_tree,
                "max_bounces": scene.cycles.max_bounces,
            },
        },
        "objects": object_summary(scene),
        "notes": notes,
        "timings_seconds": {"build": round(build_seconds, 3), "render": None if render_seconds is None else round(render_seconds, 3)},
    }
    (MANIFESTS_DIR / f"{variant}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_variant(
    variant: str,
    scene: bpy.types.Scene,
    objects: list[bpy.types.Object],
    notes: list[str],
    render: bool,
    build_started: float,
) -> None:
    blend_path = BLENDS_DIR / f"{variant}.blend"
    render_path = RENDERS_DIR / f"{variant}.png"
    remove_other_scenes(scene)
    expected_names = {obj.name for obj in objects}
    assert_unpacked(scene, expected_names)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_path),
        check_existing=False,
        relative_remap=True,
        compress=True,
    )
    elapsed_render = None
    if render:
        scene.render.filepath = str(render_path)
        started = time.perf_counter()
        bpy.ops.render.render(write_still=True, scene=scene.name)
        elapsed_render = time.perf_counter() - started
        bpy.ops.wm.save_as_mainfile(
            filepath=str(blend_path),
            check_existing=False,
            relative_remap=True,
            compress=True,
        )
    save_manifest(variant, scene, blend_path, render_path, time.perf_counter() - build_started, elapsed_render, notes)


def build_framed_variant(variant: str, render: bool) -> None:
    baseline = import_baseline()
    reset_factory()
    build_started = time.perf_counter()
    selfie, hdri = load_shared_assets()
    baseline.make_walnut_material()
    baseline.make_portrait_material(selfie)
    if variant == "01_soft_staged":
        baseline.make_plaster_material()
    portrait = baseline.make_framed_portrait_mesh()
    camera = baseline.make_camera()
    world = baseline.make_world(hdri)
    objects = [portrait, camera]
    notes = ["Shared validated framed portrait geometry and camera controls."]
    if variant == "01_soft_staged":
        wall = baseline.make_wall()
        area = baseline.make_area_light()
        objects.extend((wall, area))
        scene_name = "Soft_Staged"
        notes.append("Warm plaster wall and one large Area light.")
    else:
        scene_name = "HDRI_Only"
        notes.append("Exactly one FramedPortrait mesh plus camera; HDRI world only.")
    unlink_from_temporary_scene(objects)
    scene = link_scene(scene_name, objects, camera, world)
    if variant == "02_hdri_only":
        assert sorted(obj.type for obj in scene_objects(scene)) == ["CAMERA", "MESH"]
    save_variant(variant, scene, objects, notes, render, build_started)


def grid_values(start: float, end: float, steps: int) -> list[float]:
    return [start + (end - start) * i / steps for i in range(steps + 1)]


def make_instant_print_materials(image: bpy.types.Image) -> tuple[bpy.types.Material, bpy.types.Material, bpy.types.Material]:
    paper = bpy.data.materials.new("InstantPaperWhite")
    paper.use_nodes = True
    paper_nodes = paper.node_tree.nodes
    paper_links = paper.node_tree.links
    paper_shader = paper_nodes.get("Principled BSDF")
    assert paper_shader is not None
    paper_shader.inputs["Base Color"].default_value = (0.93, 0.89, 0.79, 1.0)
    paper_shader.inputs["Roughness"].default_value = 0.74
    paper_texcoord = paper_nodes.new("ShaderNodeTexCoord")
    paper_noise = paper_nodes.new("ShaderNodeTexNoise")
    paper_noise.inputs["Scale"].default_value = 95.0
    paper_noise.inputs["Detail"].default_value = 2.0
    paper_bump = paper_nodes.new("ShaderNodeBump")
    paper_bump.inputs["Strength"].default_value = 0.08
    paper_bump.inputs["Distance"].default_value = 0.0008
    paper_links.new(paper_texcoord.outputs["Generated"], paper_noise.inputs["Vector"])
    paper_links.new(paper_noise.outputs["Fac"], paper_bump.inputs["Height"])
    paper_links.new(paper_bump.outputs["Normal"], paper_shader.inputs["Normal"])

    emulsion = bpy.data.materials.new("InstantGlossyEmulsion")
    emulsion.use_nodes = True
    nodes = emulsion.node_tree.nodes
    links = emulsion.node_tree.links
    shader = nodes.get("Principled BSDF")
    assert shader is not None
    shader.inputs["Metallic"].default_value = 0.0
    shader.inputs["Coat Weight"].default_value = 0.30
    shader.inputs["Coat Roughness"].default_value = 0.18
    shader.inputs["Coat IOR"].default_value = 1.45
    shader.inputs["Sheen Weight"].default_value = 0.03
    texcoord = nodes.new("ShaderNodeTexCoord")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = image
    image_node.interpolation = "Linear"
    image_node.extension = "CLIP"
    image_node.location = (-500, 80)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 260.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.72
    noise.location = (-500, -160)
    roughness = nodes.new("ShaderNodeValToRGB")
    roughness.color_ramp.elements[0].color = (0.17, 0.17, 0.17, 1.0)
    roughness.color_ramp.elements[1].color = (0.36, 0.36, 0.36, 1.0)
    roughness.location = (-220, -250)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.10
    bump.inputs["Distance"].default_value = 0.0007
    bump.location = (-10, -120)
    links.new(texcoord.outputs["Generated"], noise.inputs["Vector"])
    links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
    links.new(noise.outputs["Fac"], roughness.inputs["Fac"])
    links.new(roughness.outputs["Color"], shader.inputs["Roughness"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])

    background = bpy.data.materials.new("InstantWarmBlurBackground")
    background.use_nodes = True
    bg_shader = background.node_tree.nodes.get("Principled BSDF")
    assert bg_shader is not None
    bg_shader.inputs["Base Color"].default_value = (0.34, 0.17, 0.09, 1.0)
    bg_shader.inputs["Roughness"].default_value = 0.92
    return paper, emulsion, background


def add_box_mesh(name: str, dimensions: tuple[float, float, float], location: tuple[float, float, float], material: bpy.types.Material) -> bpy.types.Object:
    x, y, z = (dimension / 2.0 for dimension in dimensions)
    vertices = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_instant_print(image: bpy.types.Image, paper: bpy.types.Material, emulsion: bpy.types.Material) -> bpy.types.Object:
    card_w, card_h = 0.90, 1.20
    side, top, bottom = 0.09, 0.09, 0.20
    outer_w, outer_h = card_w + 2 * side, card_h + top + bottom
    image_left, image_right = -card_w / 2, card_w / 2
    image_bottom = -outer_h / 2 + bottom
    image_top = image_bottom + card_h
    xs = grid_values(-outer_w / 2, image_left, 4)[:-1] + grid_values(image_left, image_right, 30)[:-1] + grid_values(image_right, outer_w / 2, 4)
    ys = grid_values(-outer_h / 2, image_bottom, 7)[:-1] + grid_values(image_bottom, image_top, 36)[:-1] + grid_values(image_top, outer_h / 2, 5)
    vertices: list[tuple[float, float, float]] = []
    for y in ys:
        for x in xs:
            nx, ny = x / outer_w, y / outer_h
            z = 0.018 + 0.011 * nx * nx + 0.006 * nx * ny + 0.003 * math.sin(ny * math.pi)
            vertices.append((x, y, z))
    faces: list[tuple[int, int, int, int]] = []
    face_mats: list[int] = []
    face_uvs: list[list[tuple[float, float]]] = []
    width = len(xs)
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            x0, x1, y0, y1 = xs[xi], xs[xi + 1], ys[yi], ys[yi + 1]
            center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
            image_face = image_left <= center_x <= image_right and image_bottom <= center_y <= image_top
            a, b, c, d = yi * width + xi, yi * width + xi + 1, (yi + 1) * width + xi + 1, (yi + 1) * width + xi
            faces.append((a, b, c, d))
            face_mats.append(1 if image_face else 0)
            if image_face:
                face_uvs.append([
                    ((x0 - image_left) / card_w, (y0 - image_bottom) / card_h),
                    ((x1 - image_left) / card_w, (y0 - image_bottom) / card_h),
                    ((x1 - image_left) / card_w, (y1 - image_bottom) / card_h),
                    ((x0 - image_left) / card_w, (y1 - image_bottom) / card_h),
                ])
            else:
                face_uvs.append([(0, 0)] * 4)
    mesh = bpy.data.meshes.new("InstantPrintSubdividedMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.materials.append(paper)
    mesh.materials.append(emulsion)
    uv_layer = mesh.uv_layers.new(name="Native3x4ImageUV")
    for polygon, mat_index, uvs in zip(mesh.polygons, face_mats, face_uvs):
        polygon.material_index = mat_index
        for loop_index, uv in zip(polygon.loop_indices, uvs):
            uv_layer.data[loop_index].uv = uv
    obj = bpy.data.objects.new("InstantPrint", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(-2.4), math.radians(1.6), math.radians(-4.0))
    solidify = obj.modifiers.new("PhysicalPaperThickness", "SOLIDIFY")
    solidify.thickness = 0.004
    bevel = obj.modifiers.new("SoftPaperEdges", "BEVEL")
    bevel.width = 0.0015
    bevel.segments = 2
    return obj


def build_instant_print(render: bool) -> None:
    scene, objects, build_started = make_instant_scene("Instant_Print")
    save_variant(
        "03_instant_print",
        scene,
        objects,
        [
            "Unbranded instant-photo paper with native 3:4 image region and white border.",
            "Subdivided curl/twist geometry, glossy uneven emulsion, and paper microtexture.",
            "Warm blurred background, HDRI world, and one Area light; no hand geometry.",
        ],
        render,
        build_started,
    )


def make_instant_scene(scene_name: str) -> tuple[bpy.types.Scene, list[bpy.types.Object], float]:
    baseline = import_baseline()
    reset_factory()
    build_started = time.perf_counter()
    selfie, hdri = load_shared_assets()
    paper, emulsion, background_material = make_instant_print_materials(selfie)
    print_obj = make_instant_print(selfie, paper, emulsion)
    camera = baseline.make_camera()
    world = baseline.make_world(hdri)
    background = add_box_mesh("InstantWarmBackground", (5.0, 4.0, 0.04), (0.0, 0.0, -0.42), background_material)
    area = baseline.make_area_light()
    unlink_from_temporary_scene([print_obj, camera, background, area])
    objects = [print_obj, camera, background, area]
    scene = link_scene(scene_name, objects, camera, world)
    assert {obj.type for obj in scene_objects(scene)} == {"CAMERA", "MESH", "LIGHT"}
    return scene, objects, build_started


def configure_cycles(scene: bpy.types.Scene) -> None:
    bpy.ops.preferences.addon_enable(module="cycles")
    preferences = bpy.context.preferences.addons["cycles"].preferences
    device_name = "Apple M1 Max CPU"
    compute_backend = "CPU"
    try:
        preferences.compute_device_type = "METAL"
        preferences.refresh_devices()
        metal_devices = [device for device in preferences.devices if device.type == "METAL"]
        if metal_devices:
            for device in preferences.devices:
                device.use = device.type == "METAL"
            scene.cycles.device = "GPU"
            device_name = ", ".join(device.name for device in metal_devices)
            compute_backend = "METAL"
        else:
            scene.cycles.device = "CPU"
    except (AttributeError, TypeError, RuntimeError):
        scene.cycles.device = "CPU"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 32
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_quality = "BALANCED"
    scene.cycles.denoising_prefilter = "ACCURATE"
    scene.cycles.denoising_use_gpu = False
    scene.cycles.use_light_tree = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_bounces = 0
    view_cycles = getattr(scene.view_layers[0], "cycles", None)
    if view_cycles is not None:
        view_cycles.use_denoising = True
        if hasattr(view_cycles, "denoising_store_passes"):
            view_cycles.denoising_store_passes = True
    scene["engine_family"] = "Cycles"
    scene["cycles_samples"] = CYCLES_SAMPLES
    scene["adaptive_sampling"] = True
    scene["denoise"] = True
    scene["cycles_compute_backend"] = compute_backend
    scene["cycles_device_name"] = device_name


def build_instant_print_cycles(render: bool) -> None:
    scene, objects, build_started = make_instant_scene("Instant_Print_Cycles")
    configure_cycles(scene)
    save_variant(
        "04_instant_print_cycles",
        scene,
        objects,
        [
            "Exact instant-print geometry, materials, camera, and HDRI world from 03_instant_print.",
            "Cycles production settings: 128 samples, adaptive sampling, and denoising enabled.",
            "Unbranded instant-photo paper; no hand geometry; one Area light plus the HDRI world.",
        ],
        render,
        build_started,
    )


def main() -> None:
    args = parse_args()
    require_inputs()
    BLENDS_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.scope in ("all", "01_soft_staged"):
        build_framed_variant("01_soft_staged", args.render)
    if args.scope in ("all", "02_hdri_only"):
        build_framed_variant("02_hdri_only", args.render)
    if args.scope in ("all", "03_instant_print"):
        build_instant_print(args.render)
    if args.scope in ("all", "04_instant_print_cycles"):
        build_instant_print_cycles(args.render)
    print(json.dumps({
        "scope": args.scope,
        "blends": [str(BLENDS_DIR / f"{args.scope}.blend")] if args.scope != "all" else [str(BLENDS_DIR / name) for name in ("01_soft_staged.blend", "02_hdri_only.blend", "03_instant_print.blend", "04_instant_print_cycles.blend")],
        "renders": [str(RENDERS_DIR / f"{args.scope}.png")] if args.scope != "all" else [str(RENDERS_DIR / name) for name in ("01_soft_staged.png", "02_hdri_only.png", "03_instant_print.png", "04_instant_print_cycles.png")],
        "render_requested": args.render,
    }, indent=2))


if __name__ == "__main__":
    main()
