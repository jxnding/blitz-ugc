#!/usr/bin/env python3
"""Prepare exact-frame H3 motion references and a pose frame with ffmpeg."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameRange:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required executable is missing: {name}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    return next(stream for stream in probe["streams"] if stream.get("codec_type") == "video")


def validate_h3_length(length: int, label: str) -> None:
    if length < 5 or (length - 5) % 17:
        raise SystemExit(f"{label} has {length} frames; H3 requires 5 + 17*n")


def extract_range(source: Path, output: Path, item: FrameRange, fps: int) -> None:
    start_seconds = Fraction(item.start, fps)
    end_seconds = Fraction(item.end + 1, fps)
    video_filter = (
        f"[0:v:0]trim=start_frame={item.start}:end_frame={item.end + 1},"
        f"setpts=N/({fps}*TB)[v]"
    )
    audio_filter = (
        f"[0:a:0]atrim=start={float(start_seconds):.9f}:end={float(end_seconds):.9f},"
        "asetpts=PTS-STARTPTS[a]"
    )
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            f"{video_filter};{audio_filter}",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-r",
            str(fps),
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
            str(output),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-frames", type=int, required=True)
    parser.add_argument("--segment-a-start", type=int, default=0)
    parser.add_argument("--segment-a-end", type=int, required=True)
    parser.add_argument("--segment-b-start", type=int, required=True)
    parser.add_argument("--segment-b-end", type=int)
    parser.add_argument("--smoke-start", type=int, required=True)
    parser.add_argument("--smoke-end", type=int)
    parser.add_argument("--pose-frame", type=int, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--outfit-sheet", type=Path)
    parser.add_argument("--outfit-crop", default="366:693:378:705", help="ffmpeg crop=w:h:x:y")
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        require_tool(executable)
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source video not found: {source}")
    if args.active_frames < 1:
        raise SystemExit("--active-frames must be positive")
    active_end = args.active_frames - 1
    segment_b_end = active_end if args.segment_b_end is None else args.segment_b_end
    smoke_end = active_end if args.smoke_end is None else args.smoke_end
    ranges = [
        FrameRange("active_reference", 0, active_end),
        FrameRange("segment_a", args.segment_a_start, args.segment_a_end),
        FrameRange("segment_b", args.segment_b_start, segment_b_end),
        FrameRange("smoke_ending", args.smoke_start, smoke_end),
    ]
    for item in ranges:
        if item.start < 0 or item.end < item.start or item.end > active_end:
            raise SystemExit(f"invalid {item.name} range: {item.start}-{item.end}")
    for item in ranges[1:]:
        validate_h3_length(item.length, item.name)
    overlap = ranges[1].end - ranges[2].start + 1
    if overlap < 0:
        raise SystemExit("segment ranges do not overlap")
    if ranges[1].length + ranges[2].length - overlap != args.active_frames:
        raise SystemExit("segment lengths minus overlap do not equal active frame count")
    if not 0 <= args.pose_frame <= active_end:
        raise SystemExit("pose frame is outside the active range")

    source_probe = ffprobe(source)
    source_video = video_stream(source_probe)
    source_fps = Fraction(source_video["r_frame_rate"])
    source_frames = int(source_video.get("nb_read_frames") or source_video.get("nb_frames") or 0)
    if source_fps != args.fps:
        raise SystemExit(f"source fps is {source_fps}, expected {args.fps}")
    if source_frames and source_frames < args.active_frames:
        raise SystemExit(f"source has only {source_frames} frames")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for item in ranges:
        output = output_dir / f"{item.name}.mp4"
        extract_range(source, output, item, args.fps)
        probe = ffprobe(output)
        stream = video_stream(probe)
        observed = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
        if observed != item.length:
            raise SystemExit(f"{output.name} has {observed} frames, expected {item.length}")
        outputs[item.name] = {
            "path": output.name,
            "sha256": sha256(output),
            "frame_range": [item.start, item.end],
            "frames": observed,
            "fps": str(Fraction(stream["r_frame_rate"])),
        }

    pose_output = output_dir / f"source_frame_{args.pose_frame:04d}.png"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select=eq(n\\,{args.pose_frame})",
            "-frames:v",
            "1",
            str(pose_output),
        ]
    )
    outputs["pose_frame"] = {
        "path": pose_output.name,
        "sha256": sha256(pose_output),
        "frame": args.pose_frame,
    }

    if args.outfit_sheet:
        outfit_sheet = args.outfit_sheet.expanduser().resolve()
        if not outfit_sheet.is_file():
            raise SystemExit(f"outfit sheet not found: {outfit_sheet}")
        outfit_output = output_dir / "yenia_hoodie_panel.png"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(outfit_sheet),
                "-vf",
                f"crop={args.outfit_crop}",
                "-frames:v",
                "1",
                str(outfit_output),
            ]
        )
        outputs["outfit_panel"] = {
            "path": outfit_output.name,
            "sha256": sha256(outfit_output),
            "crop": args.outfit_crop,
        }

    manifest = {
        "schema_version": 1,
        "source": {"path": source.name, "sha256": sha256(source)},
        "rights": "user-confirmed-cleared-for-adaptation",
        "active_frames": args.active_frames,
        "fps": args.fps,
        "overlap_frames": overlap,
        "outputs": outputs,
    }
    manifest_path = output_dir / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

