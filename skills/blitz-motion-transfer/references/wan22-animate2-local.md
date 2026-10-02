# Wan2.2 Animate-2 direct-transfer reference

Read this reference for direct character-image plus driving-video jobs. This branch deliberately bypasses ViTPose, DWPose, face crops, and manual generation segments.

## Model and runtime contract

- Official model: `Wan-AI/Wan2.2-Animate-2-14B`.
- Headless runtime: WanGP pinned to `f76ef2bb848a7d6bfc94779d493b2a9f149cd683`.
- Supported deterministic INT8 routes:
  - Distilled: model type `animate2_distilled`, checkpoint `wan2.2_animate2_14B_distilled_int8_convrot.safetensors`, 10 steps, Euler, flow shift 5, guidance 1.
  - Base: model type `animate2`, checkpoint `wan2.2_animate2_14B_int8_convrot.safetensors`, 40 steps, DPM++, flow shift 5, guidance 3.
- Use exact SDPA attention and disable the Animate-2 KV cache for both routes.
- Memory: profile 3 on a 24 GB RTX 3090. Retry once with profile 4 only after a recorded profile-3 out-of-memory failure.

Do not switch between Base and Distilled, or substitute old Wan Animate, H3, or a pose-control pipeline, without an explicit recorded decision. Official ComfyUI support was pending when this route was established, so use the pinned WanGP CLI rather than constructing an unofficial ComfyUI graph.

## Direct input roles

- `image_start` with `image_prompt_type=S`: the sole character, face, hair, outfit, and scene reference.
- `video_guide` with `video_prompt_type=UV`: direct choreography, facial-expression timing, subject scale/framing, and camera-motion reference.
- A neutral gray background in an isolated driver is a segmentation placeholder. Say this explicitly in `alt_prompt` and reject gray in the output.
- Do not supply a pose video, face-crop video, source crowd RGB, preprocessing checkpoint, or generated continuation frame.

The main prompt objectively describes the character and scene without narrating the action. The driving prompt explains what temporal information transfers and what gray pixels mean. The negative prompt rejects gray matte, source appearance, extra people, identity drift, outfit or apartment changes, text, and unstable anatomy.

## Frame and window contract

Use 24 fps and dimensions divisible by 16. Frame counts and window sizes must satisfy `4*n+1`.

- Smoke: 65 frames across the hardest identity interval.
- Full 15 seconds: clone the final driving frame once, submit 361 frames as one job, use a 65-frame sliding window with the pinned handler's native one-frame overlap, then trim generated frame 360. The live schema at the pinned commit rejects larger Animate-2 overlaps; do not patch that guard merely to imitate an older continuation recipe.
- WanGP may normalize the requested dimensions to a nearby supported latent aspect bucket. Probe the raw result; when the approved framing survives, use `assemble_single.py --model-width ... --model-height ...` for one recorded, centered lossless crop before the delivery upscale. Do not resize the raw take merely to disguise a bucket mismatch.
- A WanGP internal window is not a manually chained job. Inspect neighborhoods around every internal boundary even though the user submitted one job.

Use `scripts/run_wan_animate2.py --dry-run` before generation. Its metadata proves the two direct input roles, checkpoint selection, prompt flags, seed, settings, GPU binding method, runtime, and peak memory without storing a host address or absolute user path.

## GPU and service boundary

Discover GPUs with `nvidia-smi --query-gpu=uuid,name,memory.total`. Bind the RTX 3090 by UUID through `CUDA_VISIBLE_DEVICES`; WanGP then uses `cuda:0`. Record the exact pre-existing process/service state before stopping anything.

Stop only the recorded conflicting process. Leave unrelated GPU services alone. After generation, stop the WanGP worker, verify the 3090 is free, and restore only services the user asked to restore.

## Failure gates

Reject the smoke for gray output background, source face or clothing, an extra person, identity or age drift, apartment replacement, unstable anatomy, or incorrect motion. If gray leakage is the only failure, allow one fixed-seed retry using the same cutout composited over a static crop of the approved apartment. Do not fall back to pose extraction automatically.

For the selected full take, review every frame and the internal-window neighborhoods around frames 60-69, 120-129, 180-189, 240-249, and 300-309. Discard generated audio, restore only the approved source range, and verify decoded source/master PCM equality.
