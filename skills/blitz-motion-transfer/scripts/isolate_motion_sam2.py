#!/usr/bin/env python3
"""Track two sequential motion subjects with SAM 2.1 and place one cutout on a neutral matte."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def parse_box(value: str) -> list[float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise argparse.ArgumentTypeError("box must be x_min,y_min,x_max,y_max")
    return parts


def parse_point(value: str) -> list[float]:
    parts = [float(item) for item in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must be x,y")
    return parts


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(item) for item in value.split(",")]
    if len(parts) != 3 or any(item < 0 or item > 255 for item in parts):
        raise argparse.ArgumentTypeError("matte must be r,g,b with values from 0 to 255")
    return tuple(parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def extract_frames(
    ffmpeg: str,
    source: Path,
    frame_dir: Path,
    start: int,
    end: int,
    fps: int,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"trim=start_frame={start}:end_frame={end + 1},setpts=N/({fps}*TB)",
            "-start_number",
            "0",
            "-fps_mode",
            "passthrough",
            "-q:v",
            "2",
            str(frame_dir / "%06d.jpg"),
        ]
    )
    frames = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    expected = end - start + 1
    if len(frames) != expected:
        raise RuntimeError(f"extracted {len(frames)} frames for {start}-{end}, expected {expected}")
    return frames


def composite_mask(
    frame_path: Path,
    mask: Any,
    output_path: Path,
    matte: tuple[int, int, int],
    feather: float,
) -> int:
    import numpy as np
    from PIL import Image, ImageFilter

    frame = Image.open(frame_path).convert("RGB")
    mask_array = np.asarray(mask).squeeze().astype(bool)
    if mask_array.shape != (frame.height, frame.width):
        raise RuntimeError(f"mask shape {mask_array.shape} does not match frame {(frame.height, frame.width)}")
    alpha = Image.fromarray(mask_array.astype("uint8") * 255, mode="L")
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    background = Image.new("RGB", frame.size, matte)
    output = Image.composite(frame, background, alpha)
    output.save(output_path, quality=96, subsampling=0)
    return int(mask_array.sum())


def track_clip(
    *,
    predictor: Any,
    frame_dir: Path,
    output_dir: Path,
    global_start: int,
    box: list[float],
    positive: list[list[float]],
    negative: list[list[float]],
    matte: tuple[int, int, int],
    feather: float,
    offload_state_to_cpu: bool,
) -> list[int]:
    import numpy as np
    import torch

    frame_paths = sorted(frame_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    inference_state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=True,
    )
    points = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    kwargs: dict[str, Any] = {
        "inference_state": inference_state,
        "frame_idx": 0,
        "obj_id": 1,
        "box": np.asarray(box, dtype=np.float32),
    }
    if points:
        kwargs["points"] = np.asarray(points, dtype=np.float32)
        kwargs["labels"] = np.asarray(labels, dtype=np.int32)
    predictor.add_new_points_or_box(**kwargs)

    mask_areas: list[int] = []
    seen: set[int] = set()
    for relative_frame, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
        if 1 not in object_ids:
            raise RuntimeError(f"tracked object missing at relative frame {relative_frame}")
        object_index = list(object_ids).index(1)
        mask = (mask_logits[object_index] > 0.0).detach().cpu().numpy()
        global_frame = global_start + relative_frame
        area = composite_mask(
            frame_paths[relative_frame],
            mask,
            output_dir / f"{global_frame:06d}.jpg",
            matte,
            feather,
        )
        mask_areas.append(area)
        seen.add(relative_frame)
    if seen != set(range(len(frame_paths))):
        missing = sorted(set(range(len(frame_paths))) - seen)
        raise RuntimeError(f"SAM propagation omitted frames: {missing[:10]}")
    predictor.reset_state(inference_state)
    del inference_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mask_areas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="exact active-frame clip")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--sam2-commit", required=True)
    parser.add_argument("--active-frames", type=int, required=True)
    parser.add_argument("--handoff-frame", type=int, required=True)
    parser.add_argument("--lead-box", type=parse_box, required=True)
    parser.add_argument("--handoff-box", type=parse_box, required=True)
    parser.add_argument("--lead-positive", type=parse_point, action="append", default=[])
    parser.add_argument("--lead-negative", type=parse_point, action="append", default=[])
    parser.add_argument("--handoff-positive", type=parse_point, action="append", default=[])
    parser.add_argument("--handoff-negative", type=parse_point, action="append", default=[])
    parser.add_argument("--matte", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument("--feather", type=float, default=1.25)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable name or path")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    args = parser.parse_args()

    ffmpeg_path = shutil.which(args.ffmpeg)
    if ffmpeg_path is None and Path(args.ffmpeg).is_file():
        ffmpeg_path = str(Path(args.ffmpeg).resolve())
    if ffmpeg_path is None:
        raise SystemExit(f"required FFmpeg executable is missing: {args.ffmpeg}")
    source = args.source.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not source.is_file() or not checkpoint.is_file():
        raise SystemExit("source or checkpoint is missing")
    if not 1 <= args.handoff_frame < args.active_frames:
        raise SystemExit("handoff frame must be inside the active range")
    if args.feather < 0:
        raise SystemExit("feather must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    composite_dir = output_dir / "composite_frames"
    composite_dir.mkdir(parents=True, exist_ok=True)
    lead_frames = extract_frames(
        ffmpeg_path,
        source,
        output_dir / "lead_frames",
        0,
        args.handoff_frame - 1,
        args.fps,
    )
    handoff_frames = extract_frames(
        ffmpeg_path,
        source,
        output_dir / "handoff_frames",
        args.handoff_frame,
        args.active_frames - 1,
        args.fps,
    )

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAM 2 fallback requires the selected CUDA GPU")
    predictor = build_sam2_video_predictor(args.model_config, str(checkpoint), device=device)
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_context:
        lead_areas = track_clip(
            predictor=predictor,
            frame_dir=output_dir / "lead_frames",
            output_dir=composite_dir,
            global_start=0,
            box=args.lead_box,
            positive=args.lead_positive,
            negative=args.lead_negative,
            matte=args.matte,
            feather=args.feather,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )
        handoff_areas = track_clip(
            predictor=predictor,
            frame_dir=output_dir / "handoff_frames",
            output_dir=composite_dir,
            global_start=args.handoff_frame,
            box=args.handoff_box,
            positive=args.handoff_positive,
            negative=args.handoff_negative,
            matte=args.matte,
            feather=args.feather,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )

    observed_frames = sorted(composite_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if len(observed_frames) != args.active_frames:
        raise RuntimeError(f"composited {len(observed_frames)} frames, expected {args.active_frames}")
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    duration = args.active_frames / args.fps
    run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(args.fps),
            "-start_number",
            "0",
            "-i",
            str(composite_dir / "%06d.jpg"),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            str(args.active_frames),
            "-af",
            f"atrim=start=0:end={duration:.9f},asetpts=PTS-STARTPTS",
            "-r",
            str(args.fps),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(args.output_video),
        ]
    )
    areas = lead_areas + handoff_areas
    metadata = {
        "schema_version": 1,
        "source": {"path": source.name, "sha256": sha256(source)},
        "output": {"path": args.output_video.name, "sha256": sha256(args.output_video)},
        "sam2": {
            "repository_commit": args.sam2_commit,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
            "model_config": args.model_config,
        },
        "settings": {
            "active_frames": args.active_frames,
            "fps": args.fps,
            "lead_range": [0, args.handoff_frame - 1],
            "handoff_range": [args.handoff_frame, args.active_frames - 1],
            "lead_box": args.lead_box,
            "handoff_box": args.handoff_box,
            "lead_positive": args.lead_positive,
            "lead_negative": args.lead_negative,
            "handoff_positive": args.handoff_positive,
            "handoff_negative": args.handoff_negative,
            "matte_rgb": list(args.matte),
            "feather_pixels": args.feather,
        },
        "mask_area_pixels": {
            "minimum": min(areas),
            "maximum": max(areas),
            "empty_frames": sum(area == 0 for area in areas),
        },
    }
    metadata_path = output_dir / "sam2_isolation.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
