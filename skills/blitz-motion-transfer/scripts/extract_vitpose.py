#!/usr/bin/env python3
"""Extract Wan's official ViTPose-H pose map and optional RGB face crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def center_crop_resize(frame, width: int, height: int, cv2):
    source_h, source_w = frame.shape[:2]
    target_ratio = width / height
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        crop_w = int(round(source_h * target_ratio))
        x0 = max(0, (source_w - crop_w) // 2)
        frame = frame[:, x0:x0 + crop_w]
    else:
        crop_h = int(round(source_w / target_ratio))
        y0 = max(0, (source_h - crop_h) // 2)
        frame = frame[y0:y0 + crop_h]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def encode_sequence(frame_dir: Path, output: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("ffmpeg was not found in PATH or imageio_ffmpeg") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-start_number",
            "0",
            "-i",
            str(frame_dir / "%06d.png"),
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv444p",
            str(output),
        ],
        check=True,
    )


def face_bbox(keypoints, width: int, height: int, scale: float = 1.3):
    import numpy as np  # type: ignore

    points = keypoints.copy()[1:, :2] * (width, height)
    if not np.isfinite(points).all() or len(points) == 0:
        return None
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    initial_width = max_x - min_x
    initial_height = max_y - min_y
    if initial_width < 2 or initial_height < 2:
        return None
    expanded_area = initial_width * initial_height * scale
    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))
    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4
    x1 = max(int(min_x - delta_width), 0)
    x2 = min(int(max_x + delta_width), width)
    y1 = max(int(min_y - 3 * delta_height), 0)
    y2 = min(int(max_y + delta_height), height)
    return (x1, x2, y1, y2) if x2 > x1 and y2 > y1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--pose-output", type=Path, required=True)
    parser.add_argument("--face-output", type=Path)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--wan-preprocess-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--expected-frames", type=int)
    args = parser.parse_args()

    if not args.input_video.is_file():
        raise SystemExit(f"input video not found: {args.input_video}")
    detector_path = args.checkpoint_root / "det" / "yolov10m.onnx"
    pose_path = args.checkpoint_root / "pose2d" / "vitpose_h_wholebody.onnx"
    if not detector_path.is_file() or not pose_path.is_dir():
        raise SystemExit("official detector or ViTPose-H external-data checkpoint is missing")
    if not (args.wan_preprocess_dir / "pose2d.py").is_file():
        raise SystemExit(f"invalid Wan preprocess directory: {args.wan_preprocess_dir}")

    sys.path.insert(0, str(args.wan_preprocess_dir.resolve()))
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    import torch  # type: ignore
    from human_visualization import draw_aapose_by_meta_new  # type: ignore
    from pose2d import Pose2d  # type: ignore
    from pose2d_utils import AAPoseMeta  # type: ignore

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(f"CUDAExecutionProvider unavailable: {providers}")

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.input_video}")
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(center_crop_resize(bgr, args.width, args.height, cv2))
    capture.release()
    if not frames:
        raise SystemExit("input video decoded zero frames")
    if args.expected_frames is not None and len(frames) != args.expected_frames:
        raise SystemExit(f"expected {args.expected_frames} frames, decoded {len(frames)}")

    started = time.monotonic()
    estimator = Pose2d(
        checkpoint=str(pose_path),
        detector_checkpoint=str(detector_path),
        device="cuda",
    )
    metas = estimator(frames)
    if len(metas) != len(frames):
        raise RuntimeError(f"ViTPose returned {len(metas)} poses for {len(frames)} frames")

    pose_frames = args.pose_output.parent / f"{args.pose_output.stem}-frames"
    pose_frames.mkdir(parents=True, exist_ok=True)
    face_frames = None
    if args.face_output:
        face_frames = args.face_output.parent / f"{args.face_output.stem}-frames"
        face_frames.mkdir(parents=True, exist_ok=True)
    missing_face_frames: list[int] = []
    previous_face = np.zeros((512, 512, 3), dtype=np.uint8)

    for index, (frame, meta_raw) in enumerate(zip(frames, metas)):
        pose_meta = AAPoseMeta.from_humanapi_meta(meta_raw)
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        pose_rgb = draw_aapose_by_meta_new(canvas, pose_meta)
        cv2.imwrite(str(pose_frames / f"{index:06d}.png"), cv2.cvtColor(pose_rgb, cv2.COLOR_RGB2BGR))

        if face_frames is not None:
            bbox = face_bbox(meta_raw["keypoints_face"], args.width, args.height)
            if bbox is None:
                missing_face_frames.append(index)
                crop = previous_face
            else:
                x1, x2, y1, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    missing_face_frames.append(index)
                    crop = previous_face
                else:
                    crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                    previous_face = crop
            cv2.imwrite(str(face_frames / f"{index:06d}.png"), crop)
        if (index + 1) % 24 == 0:
            print(f"rendered {index + 1} pose frames", flush=True)

    encode_sequence(pose_frames, args.pose_output, args.fps)
    if args.face_output and face_frames is not None:
        encode_sequence(face_frames, args.face_output, args.fps)

    outputs = [
        {"role": "wan_official_vitpose_h", "name": args.pose_output.name, "sha256": sha256(args.pose_output)}
    ]
    if args.face_output:
        outputs.append(
            {"role": "wan_rgb_face_crop", "name": args.face_output.name, "sha256": sha256(args.face_output)}
        )
    metadata = {
        "schema_version": 1,
        "input": {
            "name": args.input_video.name,
            "sha256": sha256(args.input_video),
            "reported_frames": source_count,
            "reported_fps": source_fps,
        },
        "outputs": outputs,
        "frames": len(frames),
        "fps": args.fps,
        "dimensions": [args.width, args.height],
        "pose_retargeting": False,
        "face_crop_missing_frames_repeated": missing_face_frames,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "runtime": {
            "cuda_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "onnxruntime_providers": providers,
        },
        "models": {
            "repository": "Wan-AI/Wan2.2-Animate-14B",
            "revision": args.model_revision,
            "detector": {"filename": detector_path.name, "sha256": sha256(detector_path)},
            "pose": {"directory": pose_path.name, "tree_sha256": tree_sha256(pose_path)},
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(frames), "outputs": outputs, "missing_face": len(missing_face_frames)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
