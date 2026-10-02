#!/usr/bin/env python3
"""Join overlap-trimmed Wan Animate chunks, restore source audio, and export delivery files."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=Path, action="append", required=True)
    parser.add_argument("--chunk-frames", type=int, action="append", required=True)
    parser.add_argument("--audio-source", type=Path, required=True)
    parser.add_argument("--output-frames", type=int, required=True)
    parser.add_argument("--master-output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--final-width", type=int, default=1080)
    parser.add_argument("--final-height", type=int, default=1920)
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        require_tool(executable)
    if len(args.chunk) != len(args.chunk_frames):
        raise SystemExit("repeat --chunk and --chunk-frames the same number of times")
    if len(args.chunk) < 2:
        raise SystemExit("at least two Wan chunks are required")
    if not args.audio_source.is_file():
        raise SystemExit(f"audio source not found: {args.audio_source}")
    for path in args.chunk:
        if not path.is_file():
            raise SystemExit(f"chunk not found: {path}")

    probes = [probe(path) for path in args.chunk]
    streams = [stream(data, "video") for data in probes]
    dimensions = {(int(item["width"]), int(item["height"])) for item in streams}
    if len(dimensions) != 1:
        raise SystemExit(f"chunks have different dimensions: {sorted(dimensions)}")
    for index, (data, expected, video) in enumerate(zip(probes, args.chunk_frames, streams)):
        observed = frame_count(data)
        if observed != expected:
            raise SystemExit(f"chunk {index} has {observed} frames, expected {expected}")
        if Fraction(video["r_frame_rate"]) != args.fps:
            raise SystemExit(f"chunk {index} does not use {args.fps} fps")

    generated_frames = sum(args.chunk_frames)
    if not 1 <= args.output_frames <= generated_frames:
        raise SystemExit("output frames must be positive and no greater than the generated total")
    duration = Fraction(args.output_frames, args.fps)
    audio_probe = probe(args.audio_source)
    audio = stream(audio_probe, "audio")
    audio_rate = int(audio.get("sample_rate") or 48000)
    source_width, source_height = next(iter(dimensions))

    trim_parts = [
        f"[{index}:v:0]trim=start_frame=0:end_frame={frames},setpts=PTS-STARTPTS[v{index}]"
        for index, frames in enumerate(args.chunk_frames)
    ]
    concat_inputs = "".join(f"[v{index}]" for index in range(len(args.chunk)))
    audio_input = len(args.chunk)
    filter_complex = ";".join(
        trim_parts
        + [
            f"{concat_inputs}concat=n={len(args.chunk)}:v=1:a=0,"
            f"trim=start_frame=0:end_frame={args.output_frames},setpts=PTS-STARTPTS,"
            f"fps={args.fps},format=yuv420p[v]",
            f"[{audio_input}:a:0]atrim=start=0:end={float(duration):.9f},asetpts=PTS-STARTPTS[aud]",
        ]
    )
    args.master_output.parent.mkdir(parents=True, exist_ok=True)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for chunk in args.chunk:
        command.extend(["-i", str(chunk)])
    command.extend(
        [
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
    run(command)

    source_pcm = decoded_pcm_sha256(args.audio_source, duration, audio_rate)
    master_pcm = decoded_pcm_sha256(args.master_output, duration, audio_rate)
    if source_pcm != master_pcm:
        raise SystemExit("model master audio differs from the exact decoded source range")

    target_ratio = Fraction(args.final_width, args.final_height)
    source_ratio = Fraction(source_width, source_height)
    if source_ratio > target_ratio:
        crop_width = math.floor(source_height * args.final_width / args.final_height / 2) * 2
        crop_height = source_height
    else:
        crop_width = source_width
        crop_height = math.floor(source_width * args.final_height / args.final_width / 2) * 2
    video_filter = (
        f"crop={crop_width}:{crop_height}:(iw-{crop_width})/2:(ih-{crop_height})/2,"
        f"scale={args.final_width}:{args.final_height}:flags=lanczos,"
        f"fps={args.fps},format=yuv420p"
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

    seam_frames = []
    cumulative = 0
    for frames in args.chunk_frames[:-1]:
        cumulative += frames
        seam_frames.append(cumulative)
    report = {
        "schema_version": 1,
        "chunks": [
            {"file": path.name, "frames": frames, "sha256": sha256(path)}
            for path, frames in zip(args.chunk, args.chunk_frames)
        ],
        "generated_frames": generated_frames,
        "output_frames": args.output_frames,
        "trimmed_generated_tail_frames": generated_frames - args.output_frames,
        "seam_frame_indices": seam_frames,
        "fps": args.fps,
        "duration_seconds": float(duration),
        "model_resolution": [source_width, source_height],
        "assembly_crop": [crop_width, crop_height],
        "final_resolution": [args.final_width, args.final_height],
        "master": {
            "file": args.master_output.name,
            "sha256": sha256(args.master_output),
        },
        "final": {"file": args.final_output.name, "sha256": sha256(args.final_output)},
        "audio": "original-cleared-source-frames-0-through-final",
        "audio_pcm_match": True,
        "audio_pcm_sha256": source_pcm,
        "audio_sample_rate_for_comparison": audio_rate,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
