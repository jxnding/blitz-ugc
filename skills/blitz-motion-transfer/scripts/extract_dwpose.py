#!/usr/bin/env python3
"""Extract single-person DWPose body/hand and face control videos.

The input should already contain one authorized motion subject. If DWPose
returns more than one pose, this script keeps only the highest-scoring body.
Missing detections reuse the previous frame and are recorded for QA.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--pose-output", type=Path, required=True)
    parser.add_argument("--face-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True,
                        help="Extracted comfyui_controlnet_aux repository root")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--detector-repo", default="yzd-v/DWPose")
    parser.add_argument("--detector-model", default="yolox_l.onnx")
    parser.add_argument("--pose-repo", default="yzd-v/DWPose")
    parser.add_argument("--pose-model", default="dw-ll_ucoco_384.onnx")
    parser.add_argument("--verbose-model", action="store_true")
    return parser.parse_args()


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
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise RuntimeError("ffmpeg was not found in PATH or imageio_ffmpeg") from exc
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:g}", "-start_number", "0",
        "-i", str(frame_dir / "%06d.png"),
        "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("width and height must be positive")
    if not args.input_video.is_file():
        raise SystemExit(f"input video not found: {args.input_video}")
    if not (args.aux_root / "src" / "custom_controlnet_aux").is_dir():
        raise SystemExit(f"invalid comfyui_controlnet_aux root: {args.aux_root}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AUX_ANNOTATOR_CKPTS_PATH"] = str(args.checkpoint_dir.resolve())
    os.environ["AUX_ORT_PROVIDERS"] = "CUDAExecutionProvider,CPUExecutionProvider"
    sys.path.insert(0, str((args.aux_root / "src").resolve()))

    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    import torch  # type: ignore
    from custom_controlnet_aux.dwpose import DwposeDetector, draw_poses  # type: ignore

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(f"CUDAExecutionProvider unavailable: {providers}")

    load_log = io.StringIO()
    with contextlib.redirect_stdout(sys.stdout if args.verbose_model else load_log):
        detector = DwposeDetector.from_pretrained(
            args.pose_repo,
            args.detector_repo,
            det_filename=args.detector_model,
            pose_filename=args.pose_model,
            torchscript_device="cuda",
        )

    pose_frames = args.pose_output.parent / f"{args.pose_output.stem}-frames"
    face_frames = args.face_output.parent / f"{args.face_output.stem}-frames"
    pose_frames.mkdir(parents=True, exist_ok=True)
    face_frames.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.input_video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    missing_frames: list[int] = []
    multi_pose_frames: list[dict[str, int]] = []
    frame_index = 0
    previous_pose = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    previous_face = np.zeros_like(previous_pose)
    started = time.monotonic()

    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        bgr = center_crop_resize(bgr, args.width, args.height, cv2)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frame_log = io.StringIO()
        try:
            with contextlib.redirect_stdout(sys.stdout if args.verbose_model else frame_log):
                poses = detector.detect_poses(rgb)
        except Exception:
            if not args.verbose_model:
                sys.stderr.write(frame_log.getvalue())
            raise

        if poses:
            if len(poses) > 1:
                multi_pose_frames.append({"frame": frame_index, "detections": len(poses)})
            selected = max(
                poses,
                key=lambda pose: float(getattr(pose.body, "total_score", 0.0)),
            )
            selected_poses = [selected]
            pose_map = draw_poses(
                selected_poses, args.height, args.width,
                draw_body=True, draw_hand=True, draw_face=False,
            )
            face_map = draw_poses(
                selected_poses, args.height, args.width,
                draw_body=False, draw_hand=False, draw_face=True,
            )
            previous_pose = pose_map
            previous_face = face_map
        else:
            missing_frames.append(frame_index)
            pose_map = previous_pose
            face_map = previous_face

        cv2.imwrite(
            str(pose_frames / f"{frame_index:06d}.png"),
            cv2.cvtColor(pose_map, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(face_frames / f"{frame_index:06d}.png"),
            cv2.cvtColor(face_map, cv2.COLOR_RGB2BGR),
        )
        frame_index += 1
        if frame_index % 24 == 0:
            elapsed = time.monotonic() - started
            print(f"processed {frame_index} frames in {elapsed:.1f}s", flush=True)

    capture.release()
    if args.expected_frames is not None and frame_index != args.expected_frames:
        raise SystemExit(
            f"expected {args.expected_frames} frames, decoded {frame_index}"
        )
    if frame_index == 0:
        raise SystemExit("input video decoded zero frames")

    encode_sequence(pose_frames, args.pose_output, args.fps)
    encode_sequence(face_frames, args.face_output, args.fps)

    model_paths = [
        args.checkpoint_dir / args.detector_repo / args.detector_model,
        args.checkpoint_dir / args.pose_repo / args.pose_model,
    ]
    metadata = {
        "input": {
            "name": args.input_video.name,
            "sha256": sha256(args.input_video),
            "reported_frames": source_count,
            "reported_fps": source_fps,
        },
        "outputs": [
            {
                "role": "pose_body_hands",
                "name": args.pose_output.name,
                "sha256": sha256(args.pose_output),
            },
            {
                "role": "face_landmarks",
                "name": args.face_output.name,
                "sha256": sha256(args.face_output),
            },
        ],
        "frames": frame_index,
        "fps": args.fps,
        "dimensions": [args.width, args.height],
        "single_person_selection": "highest body total_score per frame",
        "missing_frames_repeated_from_previous": missing_frames,
        "multi_pose_frames_reduced_to_one": multi_pose_frames,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "runtime": {
            "cuda_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "onnxruntime_providers": providers,
        },
        "models": [
            {
                "repo": args.detector_repo,
                "filename": args.detector_model,
                "sha256": sha256(path) if path.is_file() else None,
            }
            for path in model_paths[:1]
        ] + [
            {
                "repo": args.pose_repo,
                "filename": args.pose_model,
                "sha256": sha256(path) if path.is_file() else None,
            }
            for path in model_paths[1:]
        ],
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "frames": frame_index,
        "missing": len(missing_frames),
        "multi_pose": len(multi_pose_frames),
        "pose_output": str(args.pose_output),
        "face_output": str(args.face_output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
