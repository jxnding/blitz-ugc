#!/usr/bin/env python3
"""Verify delivery properties and make sampled or exhaustive contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def stream(data: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(item for item in data["streams"] if item.get("codec_type") == kind)


def optional_stream(data: dict[str, Any], kind: str) -> dict[str, Any]:
    return next((item for item in data["streams"] if item.get("codec_type") == kind), {})


def mp4_box_offsets(path: Path) -> dict[str, int]:
    """Return offsets for top-level MP4 boxes without scanning media payloads."""
    offsets: dict[str, int] = {}
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset + 8 <= file_size:
            handle.seek(offset)
            header = handle.read(8)
            if len(header) != 8:
                break
            size, raw_type = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    break
                size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                break
            box_type = raw_type.decode("ascii", errors="replace")
            offsets.setdefault(box_type, offset)
            offset += size
    return offsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-fps", type=Fraction, default=Fraction(24, 1))
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--expected-video-codec")
    parser.add_argument("--expected-audio-codec")
    parser.add_argument("--expected-pixel-format")
    parser.add_argument("--expected-audio-channels", type=int)
    parser.add_argument("--expected-audio-sample-rate", type=int)
    parser.add_argument("--expected-audio-bitrate", type=int)
    parser.add_argument("--audio-bitrate-tolerance", type=float, default=0.15)
    parser.add_argument("--require-faststart", action="store_true")
    parser.add_argument("--seam-frame", type=int, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--all-frame-dir", type=Path)
    parser.add_argument("--frames-per-sheet", type=int, default=45)
    parser.add_argument("--sheet-columns", type=int, default=9)
    parser.add_argument("--thumbnail-width", type=int, default=180)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    data = probe(args.video)
    video = stream(data, "video")
    audio = optional_stream(data, "audio")
    observed_frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    observed_fps = Fraction(video["r_frame_rate"])
    duration = Fraction(args.expected_frames, 1) / args.expected_fps
    observed_duration = float(data["format"]["duration"])
    tolerance = float(Fraction(1, 1) / args.expected_fps)
    audio_start = float(audio.get("start_time", "0")) if audio else None
    audio_duration = float(audio.get("duration", data["format"]["duration"])) if audio else None
    audio_bitrate = int(audio["bit_rate"]) if audio.get("bit_rate") else None
    boxes = mp4_box_offsets(args.video) if args.require_faststart else {}
    faststart = not args.require_faststart or (
        "moov" in boxes and "mdat" in boxes and boxes["moov"] < boxes["mdat"]
    )
    checks = {
        "frame_count": observed_frames == args.expected_frames,
        "fps": observed_fps == args.expected_fps,
        "duration_within_one_frame": abs(observed_duration - float(duration)) <= tolerance,
        "width": args.expected_width is None or int(video["width"]) == args.expected_width,
        "height": args.expected_height is None or int(video["height"]) == args.expected_height,
        "video_codec": args.expected_video_codec is None or video.get("codec_name") == args.expected_video_codec,
        "audio_codec": args.expected_audio_codec is None or audio.get("codec_name") == args.expected_audio_codec,
        "pixel_format": args.expected_pixel_format is None or video.get("pix_fmt") == args.expected_pixel_format,
        "audio_channels": args.expected_audio_channels is None or audio.get("channels") == args.expected_audio_channels,
        "audio_sample_rate": args.expected_audio_sample_rate is None
        or int(audio.get("sample_rate", 0)) == args.expected_audio_sample_rate,
        "audio_bitrate": args.expected_audio_bitrate is None
        or (
            audio_bitrate is not None
            and abs(audio_bitrate - args.expected_audio_bitrate)
            <= args.expected_audio_bitrate * args.audio_bitrate_tolerance
        ),
        "audio_present": bool(audio),
        "audio_start_within_one_frame": audio_start is not None and abs(audio_start) <= tolerance,
        "audio_duration_within_one_frame": audio_duration is not None
        and abs(audio_duration - float(duration)) <= tolerance,
        "faststart": faststart,
    }
    if not all(checks.values()):
        raise SystemExit("verification failed: " + json.dumps(checks, sort_keys=True))

    frame_indices = sorted(
        {
            0,
            args.expected_frames // 4,
            args.expected_frames // 2,
            max(0, args.seam_frame - 2),
            max(0, args.seam_frame - 1),
            args.seam_frame,
            min(args.expected_frames - 1, args.seam_frame + 1),
            min(args.expected_frames - 1, args.seam_frame + 2),
            args.expected_frames - 2,
            args.expected_frames - 1,
        }
    )
    selection = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    columns = 4
    rows = (len(frame_indices) + columns - 1) // columns
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(args.video),
            "-vf",
            f"select='{selection}',scale=270:-2:flags=lanczos,tile={columns}x{rows}:padding=8:margin=8:color=white",
            "-frames:v",
            "1",
            str(args.contact_sheet),
        ],
        check=True,
    )
    all_frame_sheets: list[dict[str, Any]] = []
    if args.all_frame_dir:
        if args.frames_per_sheet <= 0 or args.sheet_columns <= 0 or args.thumbnail_width <= 0:
            raise SystemExit("contact-sheet dimensions must be positive")
        args.all_frame_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, args.expected_frames, args.frames_per_sheet):
            end = min(args.expected_frames - 1, start + args.frames_per_sheet - 1)
            count = end - start + 1
            rows = (count + args.sheet_columns - 1) // args.sheet_columns
            sheet = args.all_frame_dir / f"{args.video.stem}_all_frames_{start:03d}_{end:03d}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(args.video),
                    "-vf",
                    (
                        f"select='between(n\\,{start}\\,{end})',"
                        f"scale={args.thumbnail_width}:-2:flags=lanczos,"
                        f"tile={args.sheet_columns}x{rows}:padding=6:margin=6:color=white"
                    ),
                    "-frames:v",
                    "1",
                    str(sheet),
                ],
                check=True,
            )
            all_frame_sheets.append({"file": sheet.name, "frames": [start, end]})
    report = {
        "schema_version": 1,
        "file": args.video.name,
        "sha256": sha256(args.video),
        "checks": checks,
        "observed": {
            "frames": observed_frames,
            "fps": str(observed_fps),
            "duration_seconds": observed_duration,
            "resolution": [int(video["width"]), int(video["height"])],
            "video_codec": video.get("codec_name"),
            "pixel_format": video.get("pix_fmt"),
            "audio_codec": audio.get("codec_name"),
            "audio_channels": audio.get("channels"),
            "audio_sample_rate": int(audio.get("sample_rate", 0)) if audio else None,
            "audio_bitrate": audio_bitrate,
            "audio_start_seconds": audio_start,
            "audio_duration_seconds": audio_duration,
            "mp4_box_offsets": boxes,
        },
        "contact_sheet": args.contact_sheet.name,
        "contact_sheet_frames": frame_indices,
        "all_frame_sheets": all_frame_sheets,
        "visual_review_required": [
            "exactly one adult Yenia in every frame",
            "Yenia face, hair, hoodie, skirt, socks, and sneakers remain stable",
            "apartment remains stable with no source crowd, exterior, text, logos, or watermark",
            "hands, feet, limbs, shadows, camera motion, and seam are natural",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
