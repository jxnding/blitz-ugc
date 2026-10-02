#!/usr/bin/env python3
"""Normalize one generated take, restore source audio, and export master/final files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required executable is missing: {name}")


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def stream(data: dict[str, Any], kind: str) -> dict[str, Any]:
    try:
        return next(item for item in data["streams"] if item.get("codec_type") == kind)
    except StopIteration as exc:
        raise SystemExit(f"input has no {kind} stream") from exc


def frame_count(data: dict[str, Any]) -> int:
    video = stream(data, "video")
    return int(video.get("nb_read_frames") or video.get("nb_frames") or 0)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def decoded_pcm_sha256(path: Path, duration: Fraction, sample_rate: int) -> str:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            f"atrim=start=0:end={float(duration):.9f},asetpts=PTS-STARTPTS",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio-source", type=Path, required=True)
    parser.add_argument("--generated-frames", type=int, required=True)
    parser.add_argument("--output-frames", type=int, required=True)
    parser.add_argument("--master-output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--model-height", type=int)
    parser.add_argument("--final-width", type=int, default=1080)
    parser.add_argument("--final-height", type=int, default=1920)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        require_tool(executable)
    for path in (args.video, args.audio_source):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    if not 1 <= args.output_frames <= args.generated_frames:
        raise SystemExit("output frames must be positive and no greater than generated frames")

    video_probe = probe(args.video)
    audio_probe = probe(args.audio_source)
    video = stream(video_probe, "video")
    audio = stream(audio_probe, "audio")
    observed_frames = frame_count(video_probe)
    if observed_frames != args.generated_frames:
        raise SystemExit(f"generated video has {observed_frames} frames, expected {args.generated_frames}")
    if Fraction(video["r_frame_rate"]) != args.fps:
        raise SystemExit("generated video does not use the requested constant frame rate")

    source_width = int(video["width"])
    source_height = int(video["height"])
    model_width = args.model_width or source_width
    model_height = args.model_height or source_height
    if model_width <= 0 or model_height <= 0 or model_width % 2 or model_height % 2:
        raise SystemExit("model dimensions must be positive even integers")
    if model_width > source_width or model_height > source_height:
        raise SystemExit("model dimensions cannot exceed the generated video dimensions")
    model_crop_x = (source_width - model_width) // 2
    model_crop_y = (source_height - model_height) // 2

    args.master_output.parent.mkdir(parents=True, exist_ok=True)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    duration = Fraction(args.output_frames, args.fps)
    sample_rate = int(audio.get("sample_rate") or 48000)
    normalize_video = (
        f"trim=start_frame=0:end_frame={args.output_frames},setpts=PTS-STARTPTS,"
        f"crop={model_width}:{model_height}:{model_crop_x}:{model_crop_y},"
        f"fps={args.fps},format=yuv420p"
    )
    filter_complex = (
        f"[0:v:0]{normalize_video}[v];"
        f"[1:a:0]atrim=start=0:end={float(duration):.9f},asetpts=PTS-STARTPTS[aud]"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.video),
            "-i",
            str(args.audio_source),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[aud]",
            "-frames:v",
            str(args.output_frames),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-c:a",
            "pcm_s16le",
            str(args.master_output),
        ]
    )

    source_pcm = decoded_pcm_sha256(args.audio_source, duration, sample_rate)
    master_pcm = decoded_pcm_sha256(args.master_output, duration, sample_rate)
    if source_pcm != master_pcm:
        raise SystemExit("model master audio differs from the exact decoded source range")

    target_ratio = Fraction(args.final_width, args.final_height)
    source_ratio = Fraction(model_width, model_height)
    if source_ratio > target_ratio:
        crop_width = math.floor(model_height * args.final_width / args.final_height / 2) * 2
        crop_height = model_height
    else:
        crop_width = model_width
        crop_height = math.floor(model_width * args.final_height / args.final_width / 2) * 2
    video_filter = (
        f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:(ih-{crop_height})/2,"
        f"scale={args.final_width}:{args.final_height}:flags=lanczos,fps={args.fps},format=yuv420p"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.master_output),
            "-vf",
            video_filter,
            "-frames:v",
            str(args.output_frames),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(args.final_output),
        ]
    )

    report = {
        "generated_frames": args.generated_frames,
        "output_frames": args.output_frames,
        "trimmed_generated_tail_frames": args.generated_frames - args.output_frames,
        "fps": args.fps,
        "duration_seconds": float(duration),
        "native_generation_resolution": [source_width, source_height],
        "model_resolution": [model_width, model_height],
        "model_crop": [model_width, model_height, model_crop_x, model_crop_y],
        "assembly_crop": [crop_width, crop_height],
        "final_resolution": [args.final_width, args.final_height],
        "master": args.master_output.name,
        "final": args.final_output.name,
        "audio": "original-cleared-source-frames-0-through-final",
        "audio_pcm_match": True,
        "audio_pcm_sha256": source_pcm,
        "audio_sample_rate_for_comparison": sample_rate,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
