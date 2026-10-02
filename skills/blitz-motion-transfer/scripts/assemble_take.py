#!/usr/bin/env python3
"""Join overlapping H3 segments, restore source audio, and export master/final files."""

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


def video_stream(data: dict[str, Any]) -> dict[str, Any]:
    return next(stream for stream in data["streams"] if stream.get("codec_type") == "video")


def audio_stream(data: dict[str, Any]) -> dict[str, Any]:
    try:
        return next(stream for stream in data["streams"] if stream.get("codec_type") == "audio")
    except StopIteration as exc:
        raise SystemExit("audio source has no audio stream") from exc


def frame_count(data: dict[str, Any]) -> int:
    stream = video_stream(data)
    return int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)


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
    parser.add_argument("--segment-a", type=Path, required=True)
    parser.add_argument("--segment-b", type=Path, required=True)
    parser.add_argument("--audio-source", type=Path, required=True)
    parser.add_argument("--master-output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--segment-a-frames", type=int, required=True)
    parser.add_argument("--segment-b-frames", type=int, required=True)
    parser.add_argument("--overlap-frames", type=int, required=True)
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--final-width", type=int, default=1080)
    parser.add_argument("--final-height", type=int, default=1920)
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        require_tool(executable)
    for path in (args.segment_a, args.segment_b, args.audio_source):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    if args.segment_a_frames + args.segment_b_frames - args.overlap_frames != args.total_frames:
        raise SystemExit("segment frame totals do not match requested final frame count")
    a_probe = probe(args.segment_a)
    b_probe = probe(args.segment_b)
    source_probe = probe(args.audio_source)
    a_stream = video_stream(a_probe)
    b_stream = video_stream(b_probe)
    if frame_count(a_probe) != args.segment_a_frames:
        raise SystemExit("Segment A frame count mismatch")
    if frame_count(b_probe) != args.segment_b_frames:
        raise SystemExit("Segment B frame count mismatch")
    if Fraction(a_stream["r_frame_rate"]) != args.fps or Fraction(b_stream["r_frame_rate"]) != args.fps:
        raise SystemExit("segments must use the requested constant frame rate")
    if (a_stream["width"], a_stream["height"]) != (b_stream["width"], b_stream["height"]):
        raise SystemExit("segments have different dimensions")

    args.master_output.parent.mkdir(parents=True, exist_ok=True)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    duration = Fraction(args.total_frames, args.fps)
    source_audio = audio_stream(source_probe)
    source_audio_rate = int(source_audio.get("sample_rate") or 48000)
    join_filter = (
        f"[0:v:0]trim=start_frame=0:end_frame={args.segment_a_frames},setpts=PTS-STARTPTS[a];"
        f"[1:v:0]trim=start_frame={args.overlap_frames}:end_frame={args.segment_b_frames},setpts=PTS-STARTPTS[b];"
        f"[a][b]concat=n=2:v=1:a=0,fps={args.fps},format=yuv420p[v];"
        f"[2:a:0]atrim=start=0:end={float(duration):.9f},asetpts=PTS-STARTPTS[aud]"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.segment_a),
            "-i",
            str(args.segment_b),
            "-i",
            str(args.audio_source),
            "-filter_complex",
            join_filter,
            "-map",
            "[v]",
            "-map",
            "[aud]",
            "-frames:v",
            str(args.total_frames),
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
    source_audio_pcm_sha256 = decoded_pcm_sha256(args.audio_source, duration, source_audio_rate)
    master_audio_pcm_sha256 = decoded_pcm_sha256(args.master_output, duration, source_audio_rate)
    if source_audio_pcm_sha256 != master_audio_pcm_sha256:
        raise SystemExit("model master audio differs from the exact decoded source range")

    source_width = int(a_stream["width"])
    source_height = int(a_stream["height"])
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
            str(args.total_frames),
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
        "segment_a_frames": args.segment_a_frames,
        "segment_b_frames": args.segment_b_frames,
        "overlap_frames_dropped_from_b": args.overlap_frames,
        "final_frames": args.total_frames,
        "fps": args.fps,
        "duration_seconds": float(duration),
        "model_resolution": [source_width, source_height],
        "assembly_crop": [crop_width, crop_height],
        "final_resolution": [args.final_width, args.final_height],
        "master": args.master_output.name,
        "final": args.final_output.name,
        "audio": "original-cleared-source-frames-0-through-final",
        "audio_pcm_match": True,
        "audio_pcm_sha256": source_audio_pcm_sha256,
        "audio_sample_rate_for_comparison": source_audio_rate,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
