# Local Wan2.2-Animate reference

Read this reference when preparing, submitting, extending, or diagnosing a native ComfyUI Wan Animate motion-transfer run.

## Control boundary

Use animation/move mode when the approved character image should define identity, wardrobe, and scene while the source supplies motion only.

- `reference_image`: the sole default appearance and apartment reference.
- `pose_video`: colored whole-body skeleton on black. It supplies choreography, timing, scale, and position without source RGB.
- `face_video`: optional 512x512 **RGB face crops** in the upstream Wan preprocessing contract. It is not a landmark map. A source-derived face video can reintroduce source identity, so omit it by default.
- `continue_motion`: generated frames from the preceding chunk. Native Wan keeps the last five when `continue_motion_max_frames=5`.
- Do not connect `background_video` or `character_mask` in move mode.

A DWPose face-landmark video is useful QA evidence but is the wrong type for `face_video`. If expression control is later tested, use a separately approved RGB face-crop input and compare it against a pose-only fixed-seed smoke before full generation.

## Preferred preprocessing

Use the official Wan ViTPose-H WholeBody extractor first:

- Detector: `process_checkpoint/det/yolov10m.onnx`
- Pose model: `process_checkpoint/pose2d/vitpose_h_wholebody.onnx/` (ONNX external-data directory)
- Model repository revision used by the proven setup: `cb93a225fbaf1ca100f54e79da8f994995b689b3`
- Official code revision used by the proven setup: `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`

Run it against a visually approved one-person guide. For a crowded source, isolate one motion subject first; the neutral matte disappears because only the extracted skeleton enters Wan. Inspect every pose frame for missing people, jumps, wrong scale, or extra skeletons. Do not enable basic pose retargeting unless the reference and driving first frame meet the upstream front-facing stretched-pose assumptions; inspect any retargeted control before generation.

`scripts/extract_dwpose.py` is the fallback when the official checkpoint cannot run. Preserve its metadata and compare its pose contact sheet to ViTPose-H before choosing.

Primary upstream sources:

- Wan Animate preprocessing guide: https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md
- Wan2.2 repository: https://github.com/Wan-Video/Wan2.2
- ComfyUI Wan Animate guide: https://docs.comfy.org/tutorials/video/wan/wan2-2-animate

## Installed model contract

The proven local native ComfyUI setup uses:

- Diffusion: `Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors`
  - SHA-256: `b1cd5a67e968e5bb38fcf84dab58f2fb551e0889fd463acb2a657259a70bbbf6`
- LightX2V LoRA: `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`
  - SHA-256: `85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3`
- Text encoder: `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - SHA-256: `c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68`
- CLIP Vision: `clip_vision_h.safetensors`
  - SHA-256: `64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161`
- VAE: `wan_2.1_vae.safetensors`
  - SHA-256: `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b`

Proven smoke settings: 480x832, 24 fps, 77 frames, six steps, CFG 1, Euler sampler, simple scheduler, sampling shift 8, denoise 1, LightX2V strength 1, pose-only. Keep width and height divisible by 16. Wan length must satisfy `length = 4*n + 1`.

## 32 GB host profile and GPU discovery

Do not trust `nvidia-smi` index order to match PyTorch. Discover both before stopping anything:

```text
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
python -c 'import torch; print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])'
```

Use the 2080 Ti for ViTPose/DWPose preprocessing and the 3090 for Wan generation when that mapping is confirmed live. Stop only the service occupying the active GPU and restore it immediately after the stage.

For a 32 GB RAM / 24 GB VRAM generation host, launch native ComfyUI on localhost with cache-free offloading:

```text
CUDA_VISIBLE_DEVICES=<torch-3090-index> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python main.py --listen 127.0.0.1 --port 8188 --cache-none --reserve-vram 1 --disable-auto-launch
```

Record the exact PID. Query `/object_info` and require `UNETLoader`, `LoraLoaderModelOnly`, `ModelSamplingSD3`, `CLIPLoader`, `CLIPTextEncode`, `VAELoader`, `CLIPVisionLoader`, `CLIPVisionEncode`, `LoadImage`, `LoadVideo`, `GetVideoComponents`, `WanAnimateToVideo`, `KSampler`, `TrimVideoLatent`, `VAEDecode`, `ImageFromBatch`, `CreateVideo`, and `SaveVideo` before queueing.

## Continuation math

For native continuation, request 77 frames per chunk and keep five generated context frames:

1. Chunk 0: offset 0, no `continue_motion`, saved output 77 frames.
2. Chunk 1: offset 72, preceding chunk as `continue_motion`, saved output 72 frames after `trim_image`.
3. Continue offsets 144, 216, 288, and so on. Each continuation saves 72 new frames.

For an exact 360-frame delivery, make a 365-frame pose guide by duplicating the last control frame five times. Generate five chunks: `77 + 72 + 72 + 72 + 72 = 365`. Concatenate saved chunks without a dissolve, then trim frames 360-364. The padded controls and their generated frames never appear in the delivery.

At every boundary, inspect at least five frames before and after the seam. Reject a face, hair, outfit, scale, lighting, background, pose, or camera reset. Keep the original character reference connected on every chunk; use `continue_motion` for temporal state rather than replacing the character reference with a drifting generated frame.

## Failure gates

- Run a 77-frame pose-only smoke across the hardest source interval before full generation.
- Reject extra people, source setting or clothing, gray matte, text, watermark, identity handoff, anatomy failure, or seam reset.
- If one chunk fails, preserve it and name the exact defect in one bounded retry. Do not reroll unrelated accepted chunks.
- Fast spins can elongate tied hair even with pose-only control. A prompt-only retry may not fix it; record the limitation rather than silently rerolling.
- Discard model audio. Assemble the exact requested frames and mux only the approved source audio, then verify decoded PCM equality, frame count, timestamps, codec, fast-start, and all-frame visual contact sheets.
- Stop only the recorded ComfyUI PID, restart the previously stopped service, and verify the original health endpoint on every exit path.
