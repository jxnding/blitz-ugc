#!/usr/bin/env python3
"""Bridge an isolated subject handoff without an empty-frame exit or re-entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def foreground_mask(frame: np.ndarray, matte: tuple[int, int, int], threshold: int) -> np.ndarray:
    distance = np.max(np.abs(frame.astype(np.int16) - np.asarray(matte, dtype=np.int16)), axis=2)
    raw = Image.fromarray((distance > threshold).astype(np.uint8) * 255, mode="L")
    closed = raw.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(radius=1.25))
    return np.asarray(closed, dtype=np.uint8)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 32)
    if not len(xs):
        raise RuntimeError("reference frame has no foreground against the requested matte")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def extract_frame(
    ffmpeg: str,
    source: Path,
    index: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    expected = width * height * 3
    if len(result.stdout) != expected:
        raise RuntimeError(f"decoded {len(result.stdout)} bytes for frame {index}, expected {expected}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def transformed_bridge_frame(
    source_frame: np.ndarray,
    source_mask: np.ndarray,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    progress: float,
    matte: tuple[int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    source_height = y1 - y0
    target_height = ty1 - ty0
    scale = (1.0 - progress) + progress * (target_height / source_height)
    crop = Image.fromarray(source_frame[y0:y1, x0:x1], mode="RGB")
    alpha = Image.fromarray(source_mask[y0:y1, x0:x1], mode="L")
    resized_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    crop = crop.resize(resized_size, Image.Resampling.LANCZOS)
    alpha = alpha.resize(resized_size, Image.Resampling.LANCZOS)
    source_center = np.asarray(((x0 + x1) / 2, (y0 + y1) / 2), dtype=float)
    target_center = np.asarray(((tx0 + tx1) / 2, (ty0 + ty1) / 2), dtype=float)
    center = (1.0 - progress) * source_center + progress * target_center
    left = round(center[0] - crop.width / 2)
    top = round(center[1] - crop.height / 2)
    canvas = Image.new("RGB", (source_frame.shape[1], source_frame.shape[0]), matte)
    canvas.paste(crop, (left, top), alpha)
    return np.asarray(canvas, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bridge-source-frame", type=int, required=True)
    parser.add_argument("--bridge-start", type=int, required=True)
    parser.add_argument("--bridge-end", type=int, required=True)
    parser.add_argument("--target-frame", type=int, required=True)
    parser.add_argument("--matte", default="128,128,128")
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    ffmpeg = shutil.which(args.ffmpeg)
    ffprobe = shutil.which(args.ffprobe)
    if not ffmpeg or not ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source video not found: {source}")
    matte_parts = tuple(int(value) for value in args.matte.split(","))
    if len(matte_parts) != 3 or any(value < 0 or value > 255 for value in matte_parts):
        raise SystemExit("--matte must be r,g,b with values from 0 to 255")
    matte = (matte_parts[0], matte_parts[1], matte_parts[2])
    if not (
        0 <= args.bridge_source_frame < args.bridge_start <= args.bridge_end < args.target_frame
    ):
        raise SystemExit("require source_frame < bridge_start <= bridge_end < target_frame")

    probe = run_json(
        [ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(source)]
    )
    video = next(item for item in probe["streams"] if item.get("codec_type") == "video")
    width, height = int(video["width"]), int(video["height"])
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    fps_num, fps_den = (int(value) for value in video["r_frame_rate"].split("/"))
    if fps_den != 1:
        raise SystemExit("constant integer frame rate is required")
    fps = fps_num
    if args.target_frame >= frames:
        raise SystemExit("target frame is outside the video")

    source_frame = extract_frame(ffmpeg, source, args.bridge_source_frame, width, height)
    target_frame = extract_frame(ffmpeg, source, args.target_frame, width, height)
    source_mask = foreground_mask(source_frame, matte, args.threshold)
    target_mask = foreground_mask(target_frame, matte, args.threshold)
    source_box = bbox(source_mask)
    target_box = bbox(target_mask)

    decoder = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    frame_bytes = width * height * 3
    bridge_count = args.bridge_end - args.bridge_start + 1
    assert decoder.stdout is not None and encoder.stdin is not None
    for index in range(frames):
        raw = decoder.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            raise RuntimeError(f"decoder stopped at frame {index}")
        if args.bridge_start <= index <= args.bridge_end:
            progress = (index - args.bridge_start + 1) / (bridge_count + 1)
            frame = transformed_bridge_frame(
                source_frame,
                source_mask,
                source_box,
                target_box,
                progress,
                matte,
            )
            encoder.stdin.write(frame.tobytes())
        else:
            encoder.stdin.write(raw)
    decoder.stdout.close()
    encoder.stdin.close()
    decoder_code = decoder.wait()
    encoder_code = encoder.wait()
    if decoder_code or encoder_code:
        raise RuntimeError(f"ffmpeg failed: decoder={decoder_code}, encoder={encoder_code}")

    metadata = {
        "schema_version": 1,
        "source": {"path": source.name, "sha256": sha256(source)},
        "output": {"path": args.output.name, "sha256": sha256(args.output)},
        "settings": {
            "fps": fps,
            "frames": frames,
            "matte_rgb": list(matte),
            "threshold": args.threshold,
            "bridge_source_frame": args.bridge_source_frame,
            "bridge_output_frames": [args.bridge_start, args.bridge_end],
            "target_frame": args.target_frame,
            "source_bbox": list(source_box),
            "target_bbox": list(target_box),
            "method": "single-cutout pose hold with eased scale and position bridge",
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
