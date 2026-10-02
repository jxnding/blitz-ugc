# Local MiniMax H3 Ref2VA reference

Read this reference when building, submitting, or diagnosing a MiniMax H3 motion-transfer run.

## Structured roles and prompt template

H3 labels references by type and ordinal. Images appear as `<Picture 1>`, `<Picture 2>`, and so on; a motion video's soundtrack is `<Audio 1>` and its frames are `<Video 1>`.

```text
<Picture 1> is the primary character and composition reference. Preserve the same original adult character's face, hair, body proportions, outfit, apartment, lighting, and vertical composition throughout.
<Picture 2> is an identity close-up of the same character. Use it only to stabilize facial identity and hair.
[<Picture 3> is the preceding segment's final frame. Continue from this exact appearance, pose, camera state, and apartment composition.]
<Video 1> is a rights-cleared temporal guide only. Follow its single foreground subject's body movement, rhythm, timing, scale, and camera motion. Do not copy any source face, body identity, clothing, crowd, location, architecture, banner, logo, text, or watermark.
<Audio 1> is the paired rights-cleared rhythm reference. Follow its beat and timing; generated audio will not be used in the final export.
Generate exactly one continuous adult character: Yenia alone in her apartment. Never add a second person, reflection-person, partial body, crowd, outdoor element, text, logo, or watermark. Keep hands, fingers, feet, legs, fabric, shadows, and occlusions realistic and temporally stable. No identity handoff, exit, cut, morph, outfit change, background replacement, or camera jump.
```

Name one observed defect in a targeted retry, for example: `Correction: the rejected take introduced a second person at the ending; keep exactly one Yenia visible in every frame, including the handoff motion.` Do not add generic retry prose.

For a high-stakes targeted retry, expand the same roles into MiniMax's full-reference sections: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, and `non_diegetic_music`. Mark Yenia and the apartment `fully_preserved`, use `<Picture 1>` as a concrete first-frame anchor, restrict `<Video 1>` to its defined temporal role, and mark `<Audio 1>` as `reference` when it supplies rhythm only. Keep the defect-specific correction explicit after those sections.

If the later segment visibly resets to the primary anchor's pose instead of continuing the overlap, reject it as a seam failure. For the single bounded retry, place the preceding segment's final frame in `<Picture 1>`, move the full appearance anchor to `<Picture 2>`, keep the identity close-up in `<Picture 3>`, and state that frames 0–4 are continuity context whose camera and pose must lead directly into frame 5. This changes reference priority without adding an unrecorded asset or relaxing the identity gate.

If that picture-only retry still leaves a visible seam, switch conditioning method instead of adding more prompt prose. Use `MiniMaxH3AddGuide` to anchor an exact generated overlap clip from the preceding segment at frame 0 of the later segment. The guide length must satisfy `5 + 17*n`; prefer 22 frames when the surrounding motion supports it. The later motion reference must begin at the same source frame as the anchored overlap, and assembly must drop exactly the anchored frames from the later output. Preserve the guide clip, its frame range, and its hash in provenance.

## Installed model contract

- Diffusion model: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video VAE: `minimax_h3_video_vae_fp16.safetensors`
- Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- Sampler: `res_multistep`
- Scheduler: `beta`
- Denoise: `1.0`
- Image sizing: `match`
- Frame rate: 24 fps

H3 lengths must satisfy `length = 5 + 17*n`. The locally documented trained range is approximately 124–362 frames. Reference videos must be 2–15 seconds and have at least five frames. Keep dimensions divisible by 32. Start with a 124-frame probe. Use 576x1024 when it completes within the agreed probe budget; otherwise use 480x832, center-crop to 468x832 at assembly, then scale to 1080x1920.

Primary documentation:

- MiniMax H3 prompt guide: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
- ComfyUI node guide: https://github.com/Comfy-Org/embedded-docs/blob/main/comfyui_embedded_docs/docs/MiniMaxH3ReferenceToVideo/en.md

## Host and service discovery

Do not save a host address or password in the repository. Receive the SSH destination at runtime and use key-based authentication or an interactive credential mechanism outside logs.

1. Record `docker ps` and `nvidia-smi` before changing state.
2. Inspect container labels or the Compose configuration to map each service to its GPU and existing health check. Never infer the mapping from container order.
3. Discover ComfyUI with the current process list, service definitions, or a bounded search for `main.py` and the four model filenames. Record its Git commit and model hashes.
4. Stop only the explicitly approved conflicting service. Confirm the other GPU service remains healthy.
5. Launch ComfyUI on localhost and the selected GPU. Save its PID and log in a timestamped run directory.
6. Query `/object_info` and require these nodes before submission: `VHS_LoadImagePath`, `VHS_LoadVideoPath`, `UNETLoader`, `CLIPLoader`, `VAELoader`, `MiniMaxH3ReferenceToVideo`, `BasicGuider`, `RandomNoise`, `KSamplerSelect`, `BasicScheduler`, `SamplerCustomAdvanced`, `VAEDecode`, `VAEDecodeAudio`, `CreateVideo`, and `SaveVideo`. Require `MiniMaxH3AddGuide` whenever an anchored overlap is requested.
7. Submit a dry-run graph and then the probe. A successful `/prompt` validation plus completed probe demonstrates that the dynamic image, video, and paired-audio slots are accepted.
8. On every exit path, stop only the ComfyUI PID started by this run, restart the previously stopped service, and wait for its original health check to succeed.

Keep live ComfyUI input paths relative to its working directory so saved workflow JSON remains portable. Save runtime logs outside the repository or under ignored run directories.

## One-person isolation fallback

Use the official Meta SAM 2 repository and a SAM 2.1 checkpoint in a separate, timestamped environment only after the direct probe visibly fails the one-person or scene gate. Preserve the rejected probe and its reason. Run `scripts/isolate_motion_sam2.py` against the exact active-frame clip with one box and optional positive/negative clicks for each authorized temporal subject. The script keeps original scale and position but removes everything outside the selected person to a neutral matte.

Inspect every isolated frame before using it as `<Video 1>`. Reject a mask that jumps to another person, includes a second body, loses major limbs, becomes empty during active movement, or changes scale/position. Refine a named prompt frame once rather than installing a larger model or repeatedly guessing. Record the SAM repository commit, checkpoint filename/hash, boxes, clicks, frame handoff, and isolated-guide hash.

If the authorized source itself contains an exit/handoff gap, preserve the unmodified isolation as evidence and make one derived correction with `scripts/stabilize_handoff.py`. Run it through `uv run --with numpy --with pillow python ...` so its two explicit dependencies stay isolated. The bridge repeats one last safe cutout pose while easing its position and scale toward the first incoming tracked pose; it must never blend two people into one frame. Record the exact bridge and target frames, review every derived frame, and rerun the fixed-seed probe before full generation.

- Official repository: https://github.com/facebookresearch/sam2
- Recommended bounded checkpoint: `sam2.1_hiera_small.pt`
- Matching config: `configs/sam2.1/sam2.1_hiera_s.yaml`
