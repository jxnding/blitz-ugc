#!/usr/bin/env python3
"""Build, validate, submit, and record a native ComfyUI Wan Animate workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath, Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_UNET = "Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors"
DEFAULT_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
DEFAULT_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
DEFAULT_CLIP_VISION = "clip_vision_h.safetensors"
DEFAULT_VAE = "wan_2.1_vae.safetensors"
REQUIRED_NODE_TYPES = {
    "UNETLoader",
    "LoraLoaderModelOnly",
    "ModelSamplingSD3",
    "CLIPLoader",
    "CLIPTextEncode",
    "VAELoader",
    "CLIPVisionLoader",
    "CLIPVisionEncode",
    "LoadImage",
    "LoadVideo",
    "GetVideoComponents",
    "WanAnimateToVideo",
    "KSampler",
    "TrimVideoLatent",
    "VAEDecode",
    "ImageFromBatch",
    "CreateVideo",
    "SaveVideo",
}


def link(node_id: str, output: int = 0) -> list[Any]:
    return [node_id, output]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_input_name(value: str) -> str:
    """Require a portable path below ComfyUI/input."""
    normalized = PurePosixPath(value.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or value.strip() in {"", "."}:
        raise ValueError(f"ComfyUI input must be a relative input filename: {value!r}")
    return normalized.as_posix()


def validate_dimensions(width: int, height: int) -> None:
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("width and height must be at least 16 and divisible by 16")


def validate_length(length: int) -> None:
    if length < 5 or (length - 1) % 4:
        raise ValueError("Wan Animate length must satisfy length = 4*n + 1")


def build_workflow(
    *,
    reference_image: str,
    pose_video: str,
    face_video: str | None,
    continue_motion_video: str | None,
    positive_prompt: str,
    negative_prompt: str,
    output_prefix: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
    sampling_shift: float,
    lora_strength: float,
    video_frame_offset: int,
    continue_motion_max_frames: int,
    unet_name: str = DEFAULT_UNET,
    lora_name: str = DEFAULT_LORA,
    clip_name: str = DEFAULT_CLIP,
    clip_vision_name: str = DEFAULT_CLIP_VISION,
    vae_name: str = DEFAULT_VAE,
) -> dict[str, Any]:
    validate_dimensions(width, height)
    validate_length(length)
    reference_image = validate_input_name(reference_image)
    pose_video = validate_input_name(pose_video)
    face_video = validate_input_name(face_video) if face_video else None
    continue_motion_video = validate_input_name(continue_motion_video) if continue_motion_video else None
    if video_frame_offset < 0:
        raise ValueError("video_frame_offset must be non-negative")
    if continue_motion_max_frames < 1 or (continue_motion_max_frames - 1) % 4:
        raise ValueError("continue_motion_max_frames must satisfy frames = 4*n + 1")
    if not positive_prompt.strip() or not negative_prompt.strip():
        raise ValueError("positive and negative prompts must be non-empty")

    workflow: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": link("1"), "lora_name": lora_name, "strength_model": lora_strength},
        },
        "3": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": link("2"), "shift": sampling_shift},
        },
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": link("4"), "text": positive_prompt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": link("4"), "text": negative_prompt},
        },
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "8": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_name}},
        "9": {"class_type": "LoadImage", "inputs": {"image": reference_image}},
        "10": {
            "class_type": "CLIPVisionEncode",
            "inputs": {"clip_vision": link("8"), "image": link("9"), "crop": "none"},
        },
        "11": {"class_type": "LoadVideo", "inputs": {"file": pose_video}},
        "12": {"class_type": "GetVideoComponents", "inputs": {"video": link("11")}},
        "20": {
            "class_type": "WanAnimateToVideo",
            "inputs": {
                "positive": link("5"),
                "negative": link("6"),
                "vae": link("7"),
                "clip_vision_output": link("10"),
                "reference_image": link("9"),
                "pose_video": link("12", 0),
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "continue_motion_max_frames": continue_motion_max_frames,
                "video_frame_offset": video_frame_offset,
            },
        },
        "21": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("3"),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "positive": link("20", 0),
                "negative": link("20", 1),
                "latent_image": link("20", 2),
                "denoise": denoise,
            },
        },
        "22": {
            "class_type": "TrimVideoLatent",
            "inputs": {"samples": link("21"), "trim_amount": link("20", 3)},
        },
        "23": {
            "class_type": "VAEDecode",
            "inputs": {"samples": link("22"), "vae": link("7")},
        },
        "24": {
            "class_type": "ImageFromBatch",
            "inputs": {"image": link("23"), "batch_index": link("20", 4), "length": 4096},
        },
        "25": {"class_type": "CreateVideo", "inputs": {"images": link("24"), "fps": 24.0}},
        "26": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": link("25"),
                "filename_prefix": output_prefix,
                "format": "mp4",
                "codec": "h264",
                "codec.encoding": "re-encode",
                "codec.encoding.crf": 18,
            },
        },
    }
    if face_video:
        workflow["13"] = {"class_type": "LoadVideo", "inputs": {"file": face_video}}
        workflow["14"] = {"class_type": "GetVideoComponents", "inputs": {"video": link("13")}}
        workflow["20"]["inputs"]["face_video"] = link("14", 0)
    if continue_motion_video:
        workflow["15"] = {"class_type": "LoadVideo", "inputs": {"file": continue_motion_video}}
        workflow["16"] = {"class_type": "GetVideoComponents", "inputs": {"video": link("15")}}
        workflow["20"]["inputs"]["continue_motion"] = link("16", 0)
    return workflow


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        body = exc.read().decode("utf-8", errors="replace") if isinstance(exc, HTTPError) else ""
        raise RuntimeError(f"ComfyUI request failed: {exc}\n{body}") from exc


def validate_live_schema(server: str) -> dict[str, Any]:
    schema = request_json(f"{server.rstrip('/')}/object_info")
    missing = sorted(REQUIRED_NODE_TYPES - set(schema))
    if missing:
        raise RuntimeError(f"ComfyUI is missing required node types: {', '.join(missing)}")
    wan = schema["WanAnimateToVideo"].get("input", {})
    serialized = json.dumps(wan)
    for token in ("reference_image", "pose_video", "face_video", "continue_motion", "video_frame_offset"):
        if token not in serialized:
            raise RuntimeError(f"live Wan schema does not expose {token}")
    return {
        "required_nodes": "present",
        "reference_image": "present",
        "pose_video": "present",
        "face_video": "present",
        "continue_motion": "present",
        "video_frame_offset": "present",
    }


def queue_and_wait(
    server: str,
    workflow: dict[str, Any],
    timeout: float,
    poll_interval: float,
) -> tuple[str, dict[str, Any]]:
    client_id = f"blitz-wan-{uuid.uuid4().hex[:12]}"
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
        time.sleep(poll_interval)
    raise TimeoutError(f"ComfyUI generation exceeded {timeout:.0f} seconds ({prompt_id})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", required=True, help="relative filename below ComfyUI/input")
    parser.add_argument("--pose-video", required=True, help="relative filename below ComfyUI/input")
    parser.add_argument("--face-video", help="optional RGB face-crop video below ComfyUI/input")
    parser.add_argument("--continue-motion-video", help="preceding generated chunk copied below ComfyUI/input")
    parser.add_argument("--positive-prompt-file", type=Path, required=True)
    parser.add_argument("--negative-prompt-file", type=Path, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--denoise", type=float, default=1.0)
    parser.add_argument("--sampling-shift", type=float, default=8.0)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--video-frame-offset", type=int, default=0)
    parser.add_argument("--continue-motion-max-frames", type=int, default=5)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--workflow-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-schema-check", action="store_true")
    args = parser.parse_args()

    positive_prompt = args.positive_prompt_file.read_text(encoding="utf-8").strip()
    negative_prompt = args.negative_prompt_file.read_text(encoding="utf-8").strip()
    workflow = build_workflow(
        reference_image=args.reference_image,
        pose_video=args.pose_video,
        face_video=args.face_video,
        continue_motion_video=args.continue_motion_video,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        output_prefix=args.output_prefix,
        width=args.width,
        height=args.height,
        length=args.length,
        seed=args.seed,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
        scheduler=args.scheduler,
        denoise=args.denoise,
        sampling_shift=args.sampling_shift,
        lora_strength=args.lora_strength,
        video_frame_offset=args.video_frame_offset,
        continue_motion_max_frames=args.continue_motion_max_frames,
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
        if not args.skip_schema_check:
            schema_check = validate_live_schema(args.server)
        if not args.dry_run:
            prompt_id, history = queue_and_wait(args.server, workflow, args.timeout, args.poll_interval)
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
            "reference_image": validate_input_name(args.reference_image),
            "pose_video": validate_input_name(args.pose_video),
            "face_video": validate_input_name(args.face_video) if args.face_video else None,
            "continue_motion_video": (
                validate_input_name(args.continue_motion_video) if args.continue_motion_video else None
            ),
            "positive_prompt_sha256": sha256_bytes((positive_prompt + "\n").encode("utf-8")),
            "negative_prompt_sha256": sha256_bytes((negative_prompt + "\n").encode("utf-8")),
        },
        "settings": {
            "width": args.width,
            "height": args.height,
            "length": args.length,
            "fps": 24,
            "seed": args.seed,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "denoise": args.denoise,
            "sampling_shift": args.sampling_shift,
            "lora_strength": args.lora_strength,
            "video_frame_offset": args.video_frame_offset,
            "continue_motion_max_frames": args.continue_motion_max_frames,
            "output_prefix": args.output_prefix,
        },
        "models": {
            "diffusion": DEFAULT_UNET,
            "lora": DEFAULT_LORA,
            "text_encoder": DEFAULT_CLIP,
            "clip_vision": DEFAULT_CLIP_VISION,
            "vae": DEFAULT_VAE,
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
