---
name: blitz-motion-transfer
description: "Create a rights-cleared, one-character motion-transfer video with Wan2.2 Animate-2 direct video control, Wan2.2-Animate pose control, or MiniMax H3 Ref2VA, using approved appearance, motion, and audio references plus deterministic preparation, bounded retries, provenance, visual QA, and GPU-service restoration. Use for local Blitz UGC dance or performance adaptations; do not use for unlicensed copying, real-person impersonation, or multi-person output."
---

# Blitz Motion Transfer

Produce one continuous original adult character from licensed inputs while transferring only approved motion, rhythm, and camera timing.

## Non-negotiable gates

- Confirm rights to the motion and audio before generation. Treat source faces, clothing, text, brands, crowds, and locations as excluded unless separately approved.
- Label every input role: appearance images define identity/wardrobe/scene; motion video defines choreography, timing, scale, and camera movement; paired audio defines rhythm only.
- Require exactly one adult character. Reject any extra person, identity or age drift, outfit/background leakage, source text or watermark, implausible anatomy, camera jump, or visible segment seam.
- Preserve source assets and unrelated work. Use timestamped run directories; never overwrite models, character sheets, or prior outputs.
- Run a deterministic smoke test before a complete take. Permit one additional attempt only for a named failed chunk with a correction tied to the observed defect. Generate multiple complete takes only when the user requests a comparison.
- Never claim a model, seed, resolution, frame range, or restoration check without a saved record proving it.

## Workflow

1. Inspect source metadata, identify the active frame range, choose a pose-matched appearance anchor, and save input hashes and rights status.
2. Use `scripts/prepare_reference.py` to create exact-frame motion ranges with paired audio and verify their frame counts before generation.
3. Build the appearance anchor from explicitly labeled references. Keep the approved character identity, outfit, and scene; use the source frame only for pose and framing.
4. Choose the model by control boundary. Use Wan Animate-2 when the user wants one direct character-image plus driving-video job without pose extraction; read [references/wan22-animate2-local.md](references/wan22-animate2-local.md). Use old Wan Animate when source RGB must be excluded through explicit pose control; read [references/wan22-animate-local.md](references/wan22-animate-local.md). Use H3 only when raw RGB reference behavior is intentional; then read [references/minimax-h3-local.md](references/minimax-h3-local.md). Never substitute models silently.
5. If the source has a crowd, handoff, or excluded setting, use `scripts/isolate_motion_sam2.py` in a separate SAM 2.1 environment to isolate one authorized motion subject before extracting pose. If a source handoff still produces an empty-frame exit, use `scripts/stabilize_handoff.py` once to bridge the last safe cutout pose to the incoming tracked pose. Do not use isolation to retain an excluded person's likeness in the output.
6. For Animate-2, run `scripts/run_wan_animate2.py --dry-run`, then a 65-frame direct-video smoke over the hardest identity interval. A neutral matte is allowed only in the isolated driving input; reject it in generated pixels. For old Wan, extract ViTPose-H controls with `scripts/extract_vitpose.py`; use DWPose through `scripts/extract_dwpose.py` only as a recorded fallback. Run `scripts/run_wan_animate.py --dry-run`, then a 77-frame pose-only smoke. Do not connect a DWPose face-landmark map to Wan's `face_video`.
7. For Animate-2, submit the full direct driver as one job and let WanGP perform its recorded internal sliding windows. For old Wan continuation, feed the preceding generated chunk as `continue_motion`, keep five continuation frames, and advance `video_frame_offset` by 72 for each 77-frame request. For H3 segmentation and exact-overlap guidance, follow its reference instead.
8. Use `scripts/assemble_wan_chunks.py` for old Wan continuation, `scripts/assemble_take.py` for two-part H3, or `scripts/assemble_single.py` for one-job Animate-2/H3 output. Hard-join or trim exact frames, replace generated audio with the approved source audio, and require decoded-PCM equality between the approved source range and model master. Use `scripts/verify_video.py` for frame, codec, audio timing/bitrate, fast-start, sampled contact sheets, and optional exhaustive sheets through `--all-frame-dir`.
9. Watch the full selected take plus start, middle, every seam, and ending checks. Record accepted/rejected takes and reasons in the manifest.
10. In a `finally`-style cleanup, stop the exact generation worker and restore only services the user requested to restore. Verify every service expected to remain running and record only the service name and result.

## Provenance and privacy

Record relative asset paths, SHA-256 hashes, prompts, seeds, frame ranges, settings, selected take, ComfyUI commit, and exact model filenames/hashes. Do not store credentials, tokens, private host addresses, or absolute user paths. Keep publishing and platform disclosure outside the workflow unless separately authorized.

## Command interfaces

- Reference preparation: `scripts/prepare_reference.py --source SOURCE --active-frames N --segment-a-end A_END --segment-b-start B_START --smoke-start SMOKE_START --pose-frame POSE --output-dir DIR`. Optional explicit end positions and outfit-sheet crop are available through `--help`.
- H3 runner: repeat `--ref-image` in prompt-label order, then provide `--motion`, optional `--with-motion-audio`, `--prompt-file`, `--length`, `--seed`, `--width`, `--height`, `--server`, `--output-prefix`, `--workflow-output`, and `--metadata-output`. For an anchored continuation, also provide `--guide-video`, `--guide-frames`, and `--guide-frame-idx`.
- Wan ViTPose extractor: `scripts/extract_vitpose.py --input-video VIDEO --pose-output POSE --metadata-output META --wan-preprocess-dir DIR --checkpoint-root DIR --model-revision SHA --width W --height H --fps 24 --expected-frames N`. Add `--face-output` only when an RGB face-crop artifact is deliberately required for comparison.
- Wan DWPose fallback: `scripts/extract_dwpose.py --input-video VIDEO --pose-output POSE --face-output FACE_LANDMARKS --metadata-output META --aux-root DIR --checkpoint-dir DIR --width W --height H --fps 24 --expected-frames N`. Treat `FACE_LANDMARKS` as diagnostics, not Wan face conditioning.
- Wan runner: provide ComfyUI-input-relative `--reference-image` and `--pose-video`, both prompt files, length, seed, dimensions, endpoint, output prefix, workflow output, and metadata output. For extension chunks, add `--continue-motion-video` and `--video-frame-offset`. `--face-video` is optional and must be an intentional RGB face-crop video.
- Animate-2 runner: provide `--character-image`, direct `--driving-video`, the three prompt files, pinned model type/checkpoint, dimensions, frames, seed, window size/overlap, profile, GPU UUID, WanGP root/Python, output directory/name, settings/metadata/log outputs, and record root. Add `--dry-run` to validate without generation.
- Wan assembly: repeat `--chunk PATH --chunk-frames N`, then provide `--audio-source`, `--output-frames`, `--master-output`, `--final-output`, and `--report-output`. It verifies every chunk, trims only the declared generated tail, and proves decoded source/master PCM equality.
- Assembly: `scripts/assemble_take.py --segment-a A --segment-b B --overlap-frames N --audio-source AUDIO --master-output MASTER --final-output FINAL` plus the three expected frame totals. It rejects mismatched frame rates, dimensions, counts, or source/master PCM.
- Single-run assembly: `scripts/assemble_single.py --video VIDEO --audio-source AUDIO --generated-frames N --output-frames N --model-width W --model-height H --master-output MASTER --final-output FINAL --report-output REPORT` trims only the requested generated tail, optionally applies one deterministic centered model-resolution crop, discards generated audio, and proves decoded source/master PCM equality.
- Verification: `scripts/verify_video.py --video FINAL --expected-frames N --seam-frame N --contact-sheet SHEET --report REPORT`, adding expected codec, dimensions, pixel format, audio channels/rate/bitrate, and `--require-faststart` for delivery MP4s.
