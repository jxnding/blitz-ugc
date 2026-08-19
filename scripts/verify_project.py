#!/usr/bin/env python3
"""Verify the unpacked Blender project and write its top-level manifest."""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import bpy


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ASSET_DIR = PROJECT_DIR / "assets"
BLENDS_DIR = PROJECT_DIR / "blends"
RENDERS_DIR = PROJECT_DIR / "renders"
MANIFESTS_DIR = PROJECT_DIR / "manifests"
OUTPUT_PATH = PROJECT_DIR / "project_manifest.json"

SELFIE_HASH = "17bfef209000a6966063d5d319896b26d8585bd51b5026d7b76c023b8b2e6e90"
HDRI_HASH = "2fc0b1c5fb241367c7f0615b99b17e3f802ca6e8da1d7796350450d574b1f18f"
EXPECTED_IMAGES = {
    "SharedSelfie_3x4": "//../assets/selfie.png",
    "SharedStudioCountryHallHDRI": "//../assets/studio_country_hall_4k.exr",
}
VARIANTS = {
    "01_soft_staged": {
        "scene": "Soft_Staged",
        "objects": {"CAMERA": 1, "LIGHT": 1, "MESH": 2},
        "engine": "EEVEE",
    },
    "02_hdri_only": {
        "scene": "HDRI_Only",
        "objects": {"CAMERA": 1, "MESH": 1},
        "engine": "EEVEE",
    },
    "03_instant_print": {
        "scene": "Instant_Print",
        "objects": {"CAMERA": 1, "LIGHT": 1, "MESH": 2},
        "engine": "EEVEE",
    },
    "04_instant_print_cycles": {
        "scene": "Instant_Print_Cycles",
        "objects": {"CAMERA": 1, "LIGHT": 1, "MESH": 2},
        "engine": "CYCLES",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path: Path) -> list[int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if len(signature) != 24 or signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Not a PNG file: {path}")
    return list(struct.unpack(">II", signature[16:24]))


def object_types(scene: bpy.types.Scene) -> dict[str, int]:
    result: dict[str, int] = {}
    for obj in scene.collection.all_objects:
        result[obj.type] = result.get(obj.type, 0) + 1
    return result


def uv_bounds(mesh: bpy.types.Mesh) -> list[float] | None:
    if not mesh.uv_layers.active or not mesh.uv_layers.active.data:
        return None
    values = [component for loop in mesh.uv_layers.active.data for component in loop.uv]
    return [round(min(values[0::2]), 6), round(min(values[1::2]), 6), round(max(values[0::2]), 6), round(max(values[1::2]), 6)]


def normalized_value(value: object) -> object:
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 7)
    try:
        return [normalized_value(component) for component in value]
    except TypeError:
        return str(value)


def mesh_sha256(mesh: bpy.types.Mesh) -> str:
    payload = {
        "vertices": [[round(component, 7) for component in vertex.co] for vertex in mesh.vertices],
        "polygons": [list(polygon.vertices) for polygon in mesh.polygons],
        "material_indices": [polygon.material_index for polygon in mesh.polygons],
        "uvs": [] if not mesh.uv_layers.active else [[round(component, 7) for component in loop.uv] for loop in mesh.uv_layers.active.data],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def material_sha256(material: bpy.types.Material) -> str:
    node_tree = material.node_tree
    if node_tree is None:
        return hashlib.sha256(material.name.encode("utf-8")).hexdigest()
    payload = {
        "nodes": [
            {
                "name": node.name,
                "type": node.bl_idname,
                "inputs": {
                    socket.name: normalized_value(socket.default_value)
                    for socket in node.inputs
                    if hasattr(socket, "default_value") and not socket.is_linked
                },
            }
            for node in sorted(node_tree.nodes, key=lambda item: item.name)
        ],
        "links": sorted(
            (link.from_node.name, link.from_socket.name, link.to_node.name, link.to_socket.name)
            for link in node_tree.links
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def instant_scene_signature(scene: bpy.types.Scene) -> dict[str, object]:
    instant = bpy.data.objects["InstantPrint"]
    background = bpy.data.objects["InstantWarmBackground"]
    light = bpy.data.objects["SoftKeyArea"]
    return {
        "instant_mesh": mesh_sha256(instant.data),
        "instant_transform": normalized_value((*instant.location, *instant.rotation_euler, *instant.scale)),
        "instant_modifiers": [
            (modifier.name, modifier.type, normalized_value(getattr(modifier, "thickness", None)), normalized_value(getattr(modifier, "width", None)))
            for modifier in instant.modifiers
        ],
        "materials": {
            name: material_sha256(bpy.data.materials[name])
            for name in ("InstantPaperWhite", "InstantGlossyEmulsion", "InstantWarmBlurBackground")
        },
        "background_mesh": mesh_sha256(background.data),
        "background_transform": normalized_value((*background.location, *background.rotation_euler, *background.scale)),
        "light": normalized_value((light.data.energy, light.data.size, *light.data.color, *light.location, *light.rotation_euler)),
    }


def verify_variant(variant: str, contract: dict[str, object]) -> dict[str, object]:
    blend_path = BLENDS_DIR / f"{variant}.blend"
    render_path = RENDERS_DIR / f"{variant}.png"
    per_variant_manifest = MANIFESTS_DIR / f"{variant}.json"
    for path in (blend_path, render_path, per_variant_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
    if len(bpy.data.scenes) != 1:
        raise AssertionError(f"{variant}: expected one scene, found {len(bpy.data.scenes)}")
    scene = bpy.data.scenes[0]
    if scene.name != contract["scene"]:
        raise AssertionError(f"{variant}: unexpected scene {scene.name}")
    if object_types(scene) != contract["objects"]:
        raise AssertionError(f"{variant}: object contract failed: {object_types(scene)}")
    if variant == "02_hdri_only" and any(obj.type == "LIGHT" for obj in scene.collection.all_objects):
        raise AssertionError("HDRI_Only contains a Light object")

    file_images = {image.name: image for image in bpy.data.images if image.source == "FILE"}
    if set(file_images) != set(EXPECTED_IMAGES):
        raise AssertionError(f"{variant}: unexpected external images {sorted(file_images)}")
    for name, expected_path in EXPECTED_IMAGES.items():
        image = file_images[name]
        if image.filepath != expected_path:
            raise AssertionError(f"{variant}: {name} path {image.filepath!r}, expected {expected_path!r}")
        if image.packed_file is not None or len(image.packed_files) != 0:
            raise AssertionError(f"{variant}: {name} is packed")
        resolved = Path(bpy.path.abspath(image.filepath)).resolve()
        expected_resolved = (ASSET_DIR / Path(expected_path).name).resolve()
        if resolved != expected_resolved or not resolved.is_file():
            raise AssertionError(f"{variant}: unresolved asset path {resolved}")
        image.reload()

    if list(file_images["SharedSelfie_3x4"].size) != [1086, 1448]:
        raise AssertionError(f"{variant}: selfie dimensions changed")
    if contract["engine"] == "EEVEE" and scene.render.engine not in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}:
        raise AssertionError(f"{variant}: not Eevee")
    if contract["engine"] == "CYCLES" and scene.render.engine != "CYCLES":
        raise AssertionError(f"{variant}: not Cycles")
    if [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage] != [2048, 2048, 100]:
        raise AssertionError(f"{variant}: wrong render dimensions")
    if contract["engine"] == "EEVEE" and scene.get("render_samples") != 64:
        raise AssertionError(f"{variant}: wrong Eevee sample count")
    if contract["engine"] == "CYCLES":
        if scene.cycles.samples != 128 or not scene.cycles.use_adaptive_sampling or not scene.cycles.use_denoising:
            raise AssertionError(f"{variant}: wrong Cycles sampling or denoising settings")
        if scene.cycles.device != "GPU" or scene.get("cycles_compute_backend") != "METAL":
            raise AssertionError(f"{variant}: Cycles Metal GPU configuration missing")
    if round(scene.camera.data.lens, 3) != 70.0 or round(scene.camera.data.dof.aperture_fstop, 3) != 5.6:
        raise AssertionError(f"{variant}: camera contract failed")
    if scene.view_settings.view_transform != "AgX" or scene.view_settings.exposure != 0.0:
        raise AssertionError(f"{variant}: color-management contract failed")
    if not scene.world or not scene.world.use_nodes:
        raise AssertionError(f"{variant}: missing HDRI world")
    environment_nodes = [node for node in scene.world.node_tree.nodes if node.type == "TEX_ENVIRONMENT"]
    if len(environment_nodes) != 1 or environment_nodes[0].image != file_images["SharedStudioCountryHallHDRI"]:
        raise AssertionError(f"{variant}: HDRI world contract failed")

    portrait_meshes = [obj.data for obj in scene.collection.all_objects if obj.type == "MESH" and obj.name in {"FramedPortrait", "InstantPrint"}]
    if len(portrait_meshes) != 1 or uv_bounds(portrait_meshes[0]) != [0.0, 0.0, 1.0, 1.0]:
        raise AssertionError(f"{variant}: portrait UVs do not cover the full source")
    if variant in {"03_instant_print", "04_instant_print_cycles"}:
        instant = bpy.data.objects.get("InstantPrint")
        if not instant or {modifier.name for modifier in instant.modifiers} != {"PhysicalPaperThickness", "SoftPaperEdges"}:
            raise AssertionError("Instant print paper modifiers are missing")
        if "InstantGlossyEmulsion" not in bpy.data.materials or "InstantPaperWhite" not in bpy.data.materials:
            raise AssertionError("Instant print materials are missing")

    render_dimensions = png_dimensions(render_path)
    if render_dimensions != [2048, 2048]:
        raise AssertionError(f"{variant}: wrong rendered PNG dimensions {render_dimensions}")
    result = {
        "scene": scene.name,
        "blend": f"blends/{blend_path.name}",
        "blend_bytes": blend_path.stat().st_size,
        "blend_sha256": sha256_file(blend_path),
        "render": f"renders/{render_path.name}",
        "render_dimensions": render_dimensions,
        "render_sha256": sha256_file(render_path),
        "manifest": f"manifests/{per_variant_manifest.name}",
        "objects": {
            "count": len(scene.collection.all_objects),
            "types": object_types(scene),
            "names": sorted(obj.name for obj in scene.collection.all_objects),
        },
        "external_image_paths": EXPECTED_IMAGES,
        "packed_assets": False,
        "portrait_uv_bounds": [0.0, 0.0, 1.0, 1.0],
        "engine": scene.render.engine,
        "render_samples": scene.cycles.samples if scene.render.engine == "CYCLES" else 64,
    }
    if variant in {"03_instant_print", "04_instant_print_cycles"}:
        result["instant_scene_signature"] = instant_scene_signature(scene)
    if scene.render.engine == "CYCLES":
        result["cycles"] = {
            "device": scene.cycles.device,
            "compute_backend": scene.get("cycles_compute_backend"),
            "device_name": scene.get("cycles_device_name"),
            "adaptive_threshold": scene.cycles.adaptive_threshold,
            "adaptive_min_samples": scene.cycles.adaptive_min_samples,
            "denoiser": scene.cycles.denoiser,
        }
    return result


def main() -> None:
    if sha256_file(ASSET_DIR / "selfie.png") != SELFIE_HASH:
        raise AssertionError("Shared selfie hash mismatch")
    if sha256_file(ASSET_DIR / "studio_country_hall_4k.exr") != HDRI_HASH:
        raise AssertionError("Shared HDRI hash mismatch")
    if list(BLENDS_DIR.glob("*.blend1")):
        raise AssertionError("Backup .blend1 files should not be present")

    verified = {variant: verify_variant(variant, contract) for variant, contract in VARIANTS.items()}
    camera_signatures = []
    for variant in VARIANTS:
        bpy.ops.wm.open_mainfile(filepath=str(BLENDS_DIR / f"{variant}.blend"), load_ui=False)
        camera = bpy.data.scenes[0].camera
        camera_signatures.append((camera.data.lens, camera.data.dof.aperture_fstop, tuple(camera.location), tuple(camera.rotation_euler)))
    if len(set(camera_signatures)) != 1:
        raise AssertionError("Camera settings differ between variants")
    if verified["03_instant_print"]["instant_scene_signature"] != verified["04_instant_print_cycles"]["instant_scene_signature"]:
        raise AssertionError("The Eevee and Cycles instant-print scene construction differs")

    comparison_path = RENDERS_DIR / "comparison.png"
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    engine_comparison_path = RENDERS_DIR / "instant_eevee_vs_cycles.png"
    if not engine_comparison_path.is_file():
        raise FileNotFoundError(engine_comparison_path)
    manifest = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blender_version": bpy.app.version_string,
        "project_layout": "unpacked shared assets with relative Blender paths",
        "shared_assets": {
            "selfie": {
                "path": "assets/selfie.png",
                "dimensions": [1086, 1448],
                "sha256": SELFIE_HASH,
                "source_filename": "ChatGPT Image Aug 18, 2026, 01_31_57 PM.png",
            },
            "hdri": {
                "path": "assets/studio_country_hall_4k.exr",
                "sha256": HDRI_HASH,
                "url": "https://dl.polyhaven.org/file/ph-assets/HDRIs/exr/4k/studio_country_hall_4k.exr",
                "license": "CC0",
            },
        },
        "common_settings": {
            "engines": ["Eevee Next", "Cycles"],
            "resolution": [2048, 2048],
            "eevee_samples": 64,
            "cycles_samples": 128,
            "camera_lens_mm": 70.0,
            "camera_f_stop": 5.6,
            "view_transform": "AgX",
            "exposure": 0.0,
            "hdri_rotation_degrees": 0.0,
            "hdri_strength": 0.7,
        },
        "variants": verified,
        "comparison": {
            "path": "renders/comparison.png",
            "dimensions": png_dimensions(comparison_path),
            "sha256": sha256_file(comparison_path),
        },
        "instant_engine_comparison": {
            "path": "renders/instant_eevee_vs_cycles.png",
            "dimensions": png_dimensions(engine_comparison_path),
            "sha256": sha256_file(engine_comparison_path),
        },
        "verification": {
            "all_blends_opened_cleanly": True,
            "all_assets_external_and_relative": True,
            "all_assets_unpacked": True,
            "all_portrait_uvs_cover_full_source": True,
            "identical_camera_across_variants": True,
            "identical_instant_scene_except_engine": True,
            "cycles_metal_gpu": True,
            "hdri_only_object_count": 2,
            "hdri_only_light_count": 0,
            "blend_backup_count": 0,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(OUTPUT_PATH), "verified": list(verified)}, indent=2))


if __name__ == "__main__":
    main()
