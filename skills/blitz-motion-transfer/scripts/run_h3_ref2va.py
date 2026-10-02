#!/usr/bin/env python3
"""Build, validate, submit, and record a MiniMax H3 Ref2VA ComfyUI workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
REQUIRED_NODE_TYPES = {
    "VHS_LoadImagePath",
    "VHS_LoadVideoPath",
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "MiniMaxH3ReferenceToVideo",
    "BasicGuider",
    "RandomNoise",
    "KSamplerSelect",
    "BasicScheduler",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
}


def link(node_id: str, output: int = 0) -> list[Any]:
    return [node_id, output]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def portable_path(value: str) -> str:
    path = Path(value)
    parts = path.parts
    if not path.is_absolute():
        return path.as_posix()
    for marker in ("input", "output"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :]).as_posix()
    return path.name


def validate_dimensions(width: int, height: int) -> None:
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("width and height must be at least 32 and divisible by 32")


def validate_length(length: int) -> None:
    if length < 5 or (length - 5) % 17:
        raise ValueError("H3 length must be 5 + 17*n frames")


def build_workflow(
    *,
    ref_images: list[str],
    motion_video: str,
    prompt: str,
    output_prefix: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    seed: int,
    with_motion_audio: bool,
    sampler: str = "res_multistep",
    scheduler: str = "beta",
    denoise: float = 1.0,
    ref_image_size: str = "match",
    guide_video: str | None = None,
    guide_frame_idx: int = 0,
    guide_frames: int | None = None,
    unet_name: str = DEFAULT_UNET,
    clip_name: str = DEFAULT_CLIP,
    video_vae_name: str = DEFAULT_VIDEO_VAE,
    audio_vae_name: str = DEFAULT_AUDIO_VAE,
) -> dict[str, Any]:
    validate_dimensions(width, height)
    validate_length(length)
    if not ref_images:
        raise ValueError("at least one --ref-image is required")
    if len(ref_images) > 9:
        raise ValueError("H3 supports at most nine reference images")
    if not 1 <= steps <= 1000:
        raise ValueError("steps must be between 1 and 1000")
    if ref_image_size not in {"match", "max"}:
        raise ValueError("ref_image_size must be match or max")
    if (guide_video is None) != (guide_frames is None):
        raise ValueError("guide_video and guide_frames must be provided together")
    if guide_frames is not None:
        validate_length(guide_frames)
        if guide_frame_idx < 0:
            raise ValueError("guide_frame_idx must be non-negative")
        if guide_frame_idx + guide_frames > length:
            raise ValueError("guide clip does not fit inside the generated video")

    workflow: dict[str, Any] = {
        "1": {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {
                "video": motion_video,
                "force_rate": 24,
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": length,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
        "6": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "audio_vae": link("5"),
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
                "ref_image_size": ref_image_size,
                "ref_videos.ref_video_0": link("1", 0),
            },
        },
        "7": {"class_type": "BasicGuider", "inputs": {"model": link("2"), "conditioning": link("6", 0)}},
        "8": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "10": {
            "class_type": "BasicScheduler",
            "inputs": {"model": link("2"), "scheduler": scheduler, "steps": steps, "denoise": denoise},
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": link("8"),
                "guider": link("7"),
                "sampler": link("9"),
                "sigmas": link("10"),
                "latent_image": link("6", 1),
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": link("11", 0), "vae": link("4")}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": link("11", 0), "vae": link("5")}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": link("12"), "fps": 24, "audio": link("13")}},
        "15": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": link("14"),
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
                "codec.encoding": "re-encode",
                "codec.encoding.crf": 18,
            },
        },
    }
    for index, image_path in enumerate(ref_images):
        node_id = str(20 + index)
        workflow[node_id] = {
            "class_type": "VHS_LoadImagePath",
            "inputs": {"image": image_path, "custom_width": width, "custom_height": height},
        }
        workflow["6"]["inputs"][f"ref_images.ref_image_{index}"] = link(node_id, 0)
    if with_motion_audio:
        workflow["6"]["inputs"]["ref_video_audios.ref_video_audio_0"] = link("1", 2)
    if guide_video is not None and guide_frames is not None:
        workflow["30"] = {
            "class_type": "VHS_LoadVideoPath",
            "inputs": {
                "video": guide_video,
                "force_rate": 24,
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": guide_frames,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        }
        workflow["31"] = {
            "class_type": "MiniMaxH3AddGuide",
            "inputs": {
                "positive": link("6", 0),
                "vae": link("4"),
                "latent": link("6", 1),
                "image": link("30", 0),
                "frame_idx": guide_frame_idx,
            },
        }
        workflow["7"]["inputs"]["conditioning"] = link("31", 0)
    return workflow


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        body = exc.read().decode("utf-8", errors="replace") if isinstance(exc, HTTPError) else ""
        raise RuntimeError(f"ComfyUI request failed: {exc}\n{body}") from exc


def validate_live_schema(server: str, require_add_guide: bool = False) -> dict[str, Any]:
    schema = request_json(f"{server.rstrip('/')}/object_info")
    required = set(REQUIRED_NODE_TYPES)
    if require_add_guide:
        required.add("MiniMaxH3AddGuide")
    missing = sorted(required - set(schema))
    if missing:
        raise RuntimeError(f"ComfyUI is missing required node types: {', '.join(missing)}")
    h3 = schema["MiniMaxH3ReferenceToVideo"].get("input", {})
    serialized = json.dumps(h3)
    for token in ("ref_images", "ref_videos", "ref_video_audios"):
        if token not in serialized:
            raise RuntimeError(f"live H3 schema does not expose {token}")
    video_outputs = schema["VHS_LoadVideoPath"].get("output", [])
    if len(video_outputs) < 3:
        raise RuntimeError("VHS_LoadVideoPath does not expose its paired audio output")
    return {
        "required_nodes": "present",
        "h3_reference_slots": "present",
        "video_audio_output": "present",
        "guide_node": "present" if require_add_guide else "not-requested",
    }


def queue_and_wait(server: str, workflow: dict[str, Any], timeout: float) -> tuple[str, dict[str, Any]]:
    client_id = f"blitz-h3-{uuid.uuid4().hex[:12]}"
    queued = request_json(f"{server.rstrip('/')}/prompt", {"prompt": workflow, "client_id": client_id})
    if queued.get("node_errors"):
        raise RuntimeError(json.dumps(queued["node_errors"], indent=2))
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request_json(f"{server.rstrip('/')}/history/{prompt_id}")
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            status_str = status.get("status_str")
            if status_str == "success" or status.get("completed"):
                return prompt_id, item
            if status_str in {"error", "failed"}:
                raise RuntimeError(json.dumps(item, indent=2))
        time.sleep(5)
    raise TimeoutError(f"ComfyUI generation exceeded {timeout:.0f} seconds ({prompt_id})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-image", action="append", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--workflow-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--with-motion-audio", action="store_true")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--sampler", default="res_multistep")
    parser.add_argument("--scheduler", default="beta")
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--ref-image-size", choices=("match", "max"), default="match")
    parser.add_argument("--guide-video", help="optional preceding generated clip to anchor at an exact frame")
    parser.add_argument("--guide-frame-idx", type=int, default=0)
    parser.add_argument("--guide-frames", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-schema-check", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise SystemExit("prompt file is empty")
    workflow = build_workflow(
        ref_images=args.ref_image,
        motion_video=args.motion,
        prompt=prompt,
        output_prefix=args.output_prefix,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
        with_motion_audio=args.with_motion_audio,
        sampler=args.sampler,
        scheduler=args.scheduler,
        denoise=args.denoise,
        ref_image_size=args.ref_image_size,
        guide_video=args.guide_video,
        guide_frame_idx=args.guide_frame_idx,
        guide_frames=args.guide_frames,
    )
    workflow_bytes = (json.dumps(workflow, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.workflow_output.parent.mkdir(parents=True, exist_ok=True)
    args.workflow_output.write_bytes(workflow_bytes)

    schema_check: dict[str, Any] | str = "not-requested"
    prompt_id: str | None = None
    outputs: dict[str, Any] = {}
    status = "dry-run"
    failure_type: str | None = None
    caught_error: Exception | None = None
    started = time.monotonic()
    try:
        if not args.dry_run:
            if not args.skip_schema_check:
                schema_check = validate_live_schema(args.server, require_add_guide=args.guide_video is not None)
            prompt_id, history = queue_and_wait(args.server, workflow, args.timeout)
            outputs = history.get("outputs", {})
            status = "success"
    except Exception as exc:  # Preserve deterministic failure metadata, then re-raise.
        status = "failed"
        failure_type = type(exc).__name__
        caught_error = exc
    elapsed = time.monotonic() - started

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "inputs": {
            "reference_images": [portable_path(path) for path in args.ref_image],
            "motion_video": portable_path(args.motion),
            "guide_clip": (
                {
                    "path": portable_path(args.guide_video),
                    "frame_idx": args.guide_frame_idx,
                    "frames": args.guide_frames,
                }
                if args.guide_video is not None
                else None
            ),
            "paired_audio": args.with_motion_audio,
            "prompt_file": portable_path(str(args.prompt_file)),
            "prompt_sha256": sha256_bytes((prompt + "\n").encode("utf-8")),
        },
        "settings": {
            "width": args.width,
            "height": args.height,
            "length": args.length,
            "fps": 24,
            "steps": args.steps,
            "seed": args.seed,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "denoise": args.denoise,
            "ref_image_size": args.ref_image_size,
            "output_prefix": args.output_prefix,
        },
        "models": {
            "diffusion": DEFAULT_UNET,
            "text_encoder": DEFAULT_CLIP,
            "video_vae": DEFAULT_VIDEO_VAE,
            "audio_vae": DEFAULT_AUDIO_VAE,
        },
        "workflow_sha256": sha256_bytes(workflow_bytes),
        "schema_check": schema_check,
        "prompt_id": prompt_id,
        "elapsed_seconds": round(elapsed, 3),
        "failure_type": failure_type,
        "outputs": outputs,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if caught_error is not None:
        raise caught_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
