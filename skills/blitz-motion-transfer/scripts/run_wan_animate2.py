#!/usr/bin/env python3
"""Build, validate, run, and record a headless WanGP Animate-2 job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


DEFAULT_MODEL_TYPE = "animate2_distilled"
DEFAULT_CHECKPOINT = "wan2.2_animate2_14B_distilled_int8_convrot.safetensors"
BASE_MODEL_TYPE = "animate2"
BASE_CHECKPOINT = "wan2.2_animate2_14B_int8_convrot.safetensors"
PINNED_WANGP_COMMIT = "f76ef2bb848a7d6bfc94779d493b2a9f149cd683"

PINNED_RECIPES: dict[tuple[str, str], dict[str, Any]] = {
    (DEFAULT_MODEL_TYPE, DEFAULT_CHECKPOINT): {
        "variant": "distilled_int8",
        "steps": 10,
        "solver": "euler",
        "flow_shift": 5.0,
        "guidance": 1.0,
    },
    (BASE_MODEL_TYPE, BASE_CHECKPOINT): {
        "variant": "base_int8",
        "steps": 40,
        "solver": "dpm++",
        "flow_shift": 5.0,
        "guidance": 3.0,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def resolve_media(value: Path, wangp_root: Path) -> Path:
    return value.resolve() if value.is_absolute() else (wangp_root / value).resolve()


def wangp_media_path(path: Path, wangp_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), wangp_root.resolve())).as_posix()


def portable_path(path: Path, record_root: Path) -> str:
    try:
        return path.resolve().relative_to(record_root.resolve()).as_posix()
    except ValueError:
        return path.name


def ffprobe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        output = run_text(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames",
                "-of",
                "json",
                str(path),
            ]
        )
        data = json.loads(output)
        if not data.get("streams"):
            raise ValueError(f"driving video has no video stream: {path}")
        return data["streams"][0]

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("video inspection requires ffprobe or PyAV") from exc
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"driving video has no video stream: {path}")
        video = container.streams.video[0]
        decoded_frames = sum(1 for _ in container.decode(video=0))
        rate = video.average_rate or video.base_rate
        if rate is None:
            raise ValueError(f"driving video frame rate is unavailable: {path}")
        rate_text = f"{rate.numerator}/{rate.denominator}"
        return {
            "width": video.width,
            "height": video.height,
            "r_frame_rate": rate_text,
            "avg_frame_rate": rate_text,
            "nb_read_frames": str(decoded_frames),
        }


def validate_preset(wangp_root: Path, model_type: str, checkpoint: str) -> dict[str, Any]:
    preset_path = wangp_root / "defaults" / f"{model_type}.json"
    if not preset_path.is_file():
        raise ValueError(f"WanGP preset does not exist: defaults/{model_type}.json")
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    urls = preset.get("model", {}).get("URLs", [])
    selected = [url for url in urls if Path(str(url).split("?", 1)[0]).name == checkpoint]
    if len(selected) != 1:
        raise ValueError(f"checkpoint {checkpoint!r} is not uniquely declared by {preset_path.name}")
    if "int8_convrot" not in checkpoint:
        raise ValueError("Animate-2 runner requires the pinned INT8 ConvRot checkpoint")
    return {
        "preset": preset_path.name,
        "model_name": preset.get("model", {}).get("name"),
        "checkpoint": checkpoint,
        "checkpoint_url": selected[0],
    }


def validate_wangp(wangp_root: Path) -> str:
    if not (wangp_root / "wgp.py").is_file():
        raise ValueError(f"WanGP root is invalid: {wangp_root}")
    commit = run_text(["git", "rev-parse", "HEAD"], cwd=wangp_root)
    if commit != PINNED_WANGP_COMMIT:
        raise ValueError(f"WanGP commit is {commit}; expected {PINNED_WANGP_COMMIT}")
    config_path = wangp_root / "wgp_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("transformer_quantization", "int8") != "int8":
            raise ValueError("WanGP transformer_quantization must be int8")
    return commit


def runtime_versions(python: Path, wangp_root: Path) -> dict[str, str | None]:
    code = (
        "import json,sys; "
        "import torch; "
        "print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,"
        "'cuda':torch.version.cuda}))"
    )
    versions = json.loads(run_text([str(python), "-c", code], cwd=wangp_root))
    versions["platform"] = platform.platform()
    return versions


def host_ram_used_mib() -> int | None:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0])
        return (values["MemTotal"] - values["MemAvailable"]) // 1024
    except (OSError, KeyError, ValueError):
        return None


def gpu_vram_used_mib(gpu_uuid: str) -> int | None:
    try:
        raw = run_text(
            [
                "nvidia-smi",
                f"--id={gpu_uuid}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()[0]
        return int(raw.strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character-image", type=Path, required=True)
    parser.add_argument("--driving-video", type=Path, required=True)
    parser.add_argument("--main-prompt-file", type=Path, required=True)
    parser.add_argument("--driving-prompt-file", type=Path, required=True)
    parser.add_argument("--negative-prompt-file", type=Path, required=True)
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=65)
    parser.add_argument("--window-overlap", type=int, default=1)
    parser.add_argument("--profile", type=int, choices=(3, 4), default=3)
    parser.add_argument("--attention", choices=("sdpa",), default="sdpa")
    parser.add_argument("--video-prompt-type", default="UV")
    parser.add_argument("--image-prompt-type", default="S")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--wangp-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--settings-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--log-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    started_at = utc_now()
    wangp_root = args.wangp_root.resolve()
    run_root = args.record_root.resolve()
    # Preserve a virtual-environment launcher path; Path.resolve() follows its
    # symlink to the base interpreter and would silently discard the venv.
    python = Path(os.path.abspath(args.python))
    character = resolve_media(args.character_image, wangp_root)
    driver = resolve_media(args.driving_video, wangp_root)
    prompt_files = [args.main_prompt_file, args.driving_prompt_file, args.negative_prompt_file]
    if not python.is_file():
        raise SystemExit(f"Python executable not found: {python}")
    for path in (character, driver, *prompt_files):
        if not path.is_file():
            raise SystemExit(f"required input not found: {path}")
    recipe = PINNED_RECIPES.get((args.model_type, args.checkpoint))
    if recipe is None:
        supported = ", ".join(
            f"{model_type}/{checkpoint}" for model_type, checkpoint in PINNED_RECIPES
        )
        raise SystemExit(f"unsupported Animate-2 recipe; expected one of: {supported}")
    requested_recipe = {
        "steps": args.steps,
        "solver": args.solver,
        "flow_shift": args.flow_shift,
        "guidance": args.guidance,
    }
    expected_recipe = {
        key: recipe[key] for key in ("steps", "solver", "flow_shift", "guidance")
    }
    if requested_recipe != expected_recipe:
        raise SystemExit(
            f"{recipe['variant']} requires its pinned sampling recipe: {expected_recipe}"
        )
    if args.video_prompt_type != "UV" or args.image_prompt_type != "S":
        raise SystemExit("Animate-2 direct transfer requires video_prompt_type=UV and image_prompt_type=S")
    if min(args.width, args.height) < 16 or args.width % 16 or args.height % 16:
        raise SystemExit("width and height must be at least 16 and divisible by 16")
    if args.frames < 5 or (args.frames - 1) % 4:
        raise SystemExit("frames must satisfy frames = 4*n + 1")
    if args.window_size < 5 or (args.window_size - 1) % 4:
        raise SystemExit("window size must satisfy size = 4*n + 1")
    if not 1 <= args.window_overlap < args.window_size:
        raise SystemExit("window overlap must be positive and smaller than the window")
    if args.window_overlap != 1:
        raise SystemExit("the pinned WanGP Animate-2 handler requires a one-frame sliding-window overlap")
    if args.fps != 24:
        raise SystemExit("this workflow is pinned to 24 fps")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", args.gpu_uuid):
        raise SystemExit("--gpu-uuid must be an NVIDIA GPU UUID")

    commit = validate_wangp(wangp_root)
    preset = validate_preset(wangp_root, args.model_type, args.checkpoint)
    driver_probe = ffprobe_video(driver)
    observed_frames = int(driver_probe.get("nb_read_frames") or driver_probe.get("nb_frames") or 0)
    observed_fps = Fraction(driver_probe.get("avg_frame_rate") or driver_probe["r_frame_rate"])
    if observed_frames != args.frames:
        raise SystemExit(f"driving video has {observed_frames} frames; expected {args.frames}")
    if observed_fps != args.fps:
        raise SystemExit(f"driving video is {observed_fps} fps; expected {args.fps}")

    main_prompt, driving_prompt, negative_prompt = (
        path.read_text(encoding="utf-8").strip() for path in prompt_files
    )
    if not all((main_prompt, driving_prompt, negative_prompt)):
        raise SystemExit("all three prompt files must be non-empty")

    settings = {
        "model_type": args.model_type,
        "prompt": main_prompt,
        "negative_prompt": negative_prompt,
        "alt_prompt": driving_prompt,
        "video_prompt_type": args.video_prompt_type,
        "image_prompt_type": args.image_prompt_type,
        "image_start": wangp_media_path(character, wangp_root),
        "video_guide": wangp_media_path(driver, wangp_root),
        "resolution": f"{args.width}x{args.height}",
        "video_length": args.frames,
        "force_fps": "control",
        "num_inference_steps": args.steps,
        "sample_solver": args.solver,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance,
        "seed": args.seed,
        "sliding_window_size": args.window_size,
        "sliding_window_overlap": args.window_overlap,
        "remove_background_images_ref": 0,
        "prompt_enhancer": "",
        "custom_settings": {"animate2_kv_cache": "Disabled"},
        "override_profile": args.profile,
        "override_attention": args.attention,
        "repeat_generation": 1,
        "output_filename": args.output_name,
    }
    write_json(args.settings_output, settings)

    checkpoint_path = wangp_root / "ckpts" / args.checkpoint
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "prepared",
        "dry_run": args.dry_run,
        "started_at": started_at,
        "input_roles": {
            "character_image": {
                "path": portable_path(character, run_root),
                "sha256": sha256(character),
                "role": "identity, face, hair, outfit, and apartment",
            },
            "driving_video": {
                "path": portable_path(driver, run_root),
                "sha256": sha256(driver),
                "role": "choreography, expression timing, framing, and camera motion only",
                "frames": observed_frames,
                "fps": str(observed_fps),
            },
        },
        "direct_inputs_only": True,
        "excluded_inputs": ["pose video", "face-crop video", "source crowd RGB", "preprocessing model"],
        "model": {
            "variant": recipe["variant"],
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path.is_file() else None,
            "preset": preset["preset"],
            "wan_gp_commit": commit,
        },
        "settings": {
            key: value
            for key, value in settings.items()
            if key not in {"image_start", "video_guide", "prompt", "negative_prompt", "alt_prompt"}
        },
        "prompts": {
            "main": main_prompt,
            "driving": driving_prompt,
            "negative": negative_prompt,
        },
        "runtime": {
            "python": portable_path(python, wangp_root),
            "versions": runtime_versions(python, wangp_root),
            "gpu_binding": "CUDA_VISIBLE_DEVICES by GPU UUID; WanGP sees cuda:0",
            "gpu_model": run_text(
                ["nvidia-smi", f"--id={args.gpu_uuid}", "--query-gpu=name", "--format=csv,noheader"]
            ),
            "attention": args.attention,
            "profile": args.profile,
        },
        "artifacts": {
            "settings": portable_path(args.settings_output, run_root),
            "log": portable_path(args.log_output, run_root),
            "output_dir": portable_path(args.output_dir, run_root),
        },
    }
    write_json(args.metadata_output, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_output.parent.mkdir(parents=True, exist_ok=True)
    prior_files = {path.resolve() for path in args.output_dir.rglob("*") if path.is_file()}
    command = [
        str(python),
        str(wangp_root / "wgp.py"),
        "--process",
        str(args.settings_output.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--profile",
        str(args.profile),
        "--attention",
        args.attention,
        "--gpu",
        "cuda:0",
        "--verbose",
        "2",
    ]
    if args.dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"

    peak_ram: int | None = None
    peak_vram: int | None = None
    return_code: int | None = None
    with args.log_output.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=wangp_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while (return_code := process.poll()) is None:
            ram = host_ram_used_mib()
            vram = gpu_vram_used_mib(args.gpu_uuid)
            peak_ram = ram if ram is not None and peak_ram is None else max(peak_ram or 0, ram or 0)
            peak_vram = vram if vram is not None and peak_vram is None else max(peak_vram or 0, vram or 0)
            time.sleep(max(1.0, args.poll_seconds))

    new_files = sorted(
        path.resolve()
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.resolve() not in prior_files
    )
    metadata["status"] = "success" if return_code == 0 else "failed"
    metadata["completed_at"] = utc_now()
    metadata["runtime"].update(
        {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_host_ram_mib": peak_ram,
            "peak_gpu_vram_mib": peak_vram,
            "exit_code": return_code,
        }
    )
    if checkpoint_path.is_file():
        metadata["model"]["checkpoint_sha256"] = sha256(checkpoint_path)
        metadata["model"]["checkpoint_bytes"] = checkpoint_path.stat().st_size
    metadata["artifacts"]["generated_files"] = [portable_path(path, run_root) for path in new_files]
    write_json(args.metadata_output, metadata)
    if return_code != 0:
        raise SystemExit(f"WanGP exited with code {return_code}; inspect {args.log_output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
