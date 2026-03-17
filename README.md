# ComfyUI-Wan22FMLF_svi-fix (SVI Boost)

> Multi-frame reference conditioning nodes for Wan2.2 A14B I2V models, with enhanced SVI (Stable Video Infinity) seamless continuation.

This fork adds the **SVI Pro Advanced** node on top of [wallen0322/ComfyUI-Wan22FMLF](https://github.com/wallen0322/ComfyUI-Wan22FMLF), fixing high-resolution motion loss, splice jumping, and adding boost controls for motion and detail.

---

## 4-in-1 Long Video Demo (24fps 720p, 403 frames)

Top: with lip sync | Bottom: without lip sync

https://github.com/user-attachments/assets/dc1cf2a4-3c6a-4210-a247-e53c2423f776

Two segments of 161 frames + one segment of 81 frames. Each segment's variation is controlled by the middle frame and prompt. Lip sync is applied uniformly at the end.

> Chorus scenes don't support lip sync well — avoid scenes with large changes in the number of people.

This example workflow combines: SVI Pro Boost video extension + FMLF free frame control + VBVR physics LoRA + InfiniTalk lip sync (Painter AV2V). The workflow file is in `example_workflows/`.

---

## SVI Pro Advanced Node

### Inputs

The node merges FMLF and SVI Pro interfaces:

- **SVI inputs:** `anchor_samples` (anchor latent), `prev_latent` (previous video segment)
- **FMLF inputs:** `start_image`, `middle_image`, `end_image` (reference frames — lower priority than `anchor_samples`, weight-dependent)

When `prev_latent` is connected, `start_image` has minimal effect.

### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `motion_influence` | 1.0 | 0.0 - 2.0 | Motion transfer weight from previous segment. Too high or too low makes splices unnatural. Low-res: increase. High-res: decrease. |
| `overlap_frames` | 4 | 0 - 128 | Number of pixel frames fed to motion influence (internally divided by 4 for latent frames). Set to 0 to disable overlap. |
| `motion_boost` | 1.0 | 1.0 - 3.0 | Amplifies motion by scaling frame-to-frame differences in the motion latent. |
| `detail_boost` | 1.0 | 1.0 - 4.0 | Increases motion dynamics by adjusting the mask decay rate and scaling how many frames are pulled from prev_latent. Higher values = faster decay = more freedom for the model. |

### Recommended Presets

**Standard (enhanced SVI defaults):**

```
motion_influence: 1.0-1.3 | overlap_frames: 4 | motion_boost: 1.0-1.5 | detail_boost: 1.0-1.5
```

> Keep all start/end frame strengths below 0.5 — higher values cause gray-out artifacts.

**Aggressive (may cause 4-frame flicker at start/end, splices may be less smooth):**

| Preset | motion_influence | overlap_frames | motion_boost | detail_boost | Seamless overlap |
|--------|-----------------|----------------|--------------|--------------|-----------------|
| High transfer + high motion | 2.0 | 16 | 2.0 | 1.0 | 17 frames |
| Low transfer + high motion | 0.7 | 4 | 2.0 | 2.0 | 5 frames |

See the SVI Pro Boost example workflow for detailed usage.

---

## Other Nodes

### Wan First-Middle-Last Frame

Generates video with 3 reference frames: start, middle, and end.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `positive` | CONDITIONING | **required** | Positive prompt conditioning |
| `negative` | CONDITIONING | **required** | Negative prompt conditioning |
| `vae` | VAE | **required** | VAE model for encoding |
| `width` | INT | 832 | Video width (multiple of 16) |
| `height` | INT | 480 | Video height (multiple of 16) |
| `length` | INT | 81 | Total frames (multiple of 4 + 1) |
| `batch_size` | INT | 1 | Number of videos to generate |
| `start_image` | IMAGE | optional | Start frame reference |
| `middle_image` | IMAGE | optional | Middle frame reference |
| `end_image` | IMAGE | optional | End frame reference |
| `middle_frame_ratio` | FLOAT | 0.5 | Middle frame position (0.0 - 1.0) |
| `high_noise_mid_strength` | FLOAT | 0.8 | High-noise middle frame strength (0 = loose, 1 = strict) |
| `low_noise_start_strength` | FLOAT | 1.0 | Low-noise start frame strength |
| `low_noise_mid_strength` | FLOAT | 0.2 | Low-noise middle frame strength |
| `low_noise_end_strength` | FLOAT | 1.0 | Low-noise end frame strength |
| `structural_repulsion_boost` | FLOAT | 1.0 | Motion enhancement (1.0 - 2.0), affects high-noise stage only |
| `clip_vision_*` | CLIP_VISION_OUTPUT | optional | CLIP Vision embeddings for start/middle/end frames |

### Wan Multi-Frame Reference

General-purpose multi-frame reference node supporting 2, 3, 4, or more reference frames with flexible positioning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_images` | IMAGE | **required** | Reference frame images |
| `ref_positions` | STRING | `""` (auto) | Frame positions: `"0,40,80"`, `"0,0.5,1.0"`, or empty for auto-distribute |
| `ref_strength` | FLOAT | 0.5 | Middle frame constraint strength (0 - 1) |
| `fade_frames` | INT | 2 | Fade-out transition frames (0 - 8) |
| `clip_vision_output` | CLIP_VISION_OUTPUT | optional | CLIP Vision output |

#### `ref_positions` Format

- **Empty string** (recommended): auto-distributes frames evenly. 3 images with length=81 gives positions 0, 40, 80.
- **Ratio values** (0.0 - 1.0): `"0, 0.25, 0.5, 0.75, 1"` — positions as percentage of video length.
- **Absolute indices** (>= 2): `"0, 20, 40, 60, 80"` — direct frame positions, auto-clipped to valid range.
- **JSON array**: `"[0, 0.25, 0.5, 60, 1.0]"` — mix of ratios and absolutes.

All positions are automatically aligned to multiples of 4 (latent alignment). Adjacent frames maintain at least 4 frames of spacing.

---

## Quick Start

### Requirements

- Use the **official model weights** — quantized models are not recommended
- **Single-person mode** helps prevent color accumulation and brightness flickering

### Recommended Resolutions

| Category | Resolutions |
|----------|------------|
| Low-res | 480x832, 832x480, 576x1024 |
| High-res | 704x1280, 1280x704 |

> **Warning:** 720x1280 causes middle-frame flickering — avoid this resolution.

### Noise Strength Tips

- **High-noise steps:** 2 steps is enough. More steps increase middle-frame flicker probability.
- **Middle frame strength:** Normal scenes: high=0.6-0.8, low=0.2. Complex scenes: high=0.6-0.8, low=0.

### Scene Tips

- **High-variation scenes** (e.g. transformations): use normal mode, reduce LightX2V LoRA weight to ~0.6. Otherwise low-noise will suppress the transitions.
- **Infinite long video with multi-image reference:** see the 3-image loop workflow with the visual image picker node.

---

## Example Workflows

Available in `example_workflows/`:

- `SVI pro boost.json` — SVI Pro Boost video extension
- `四合一无限IA2V...json` — 4-in-1 workflow (SVI + FMLF + InfiniTalk + VBVR, requires Painter Nodes)
- `Long video + segmented prompt words.json` — Long video with per-segment prompts
- `Wan22FMLF-1109update.json` — Multi-frame reference base workflow

<img alt="Example workflow screenshot" src="https://github.com/user-attachments/assets/c5560d54-87a5-4736-8bbb-78a5a8125e13" />

---

## Changelog (svi-fix)

### 2025-03-06
- Added 4-in-1 example workflow combining the full Wan2.2 ecosystem (requires Painter Nodes)

### 2025-03-05
- Fixed start/end frame flickering in SVI Advanced. Still flickers when boost > 2.
- Updated recommended parameter presets and example workflow.

### 2025-03-03
- Separated `anchor_samples` from `start_image` for better character locking.
- Disabling boost now gives vanilla SVI Pro behavior.
- Began rebasing onto upstream SVI Pro as the foundation.

### 2025-02-01
- Added `wan_svi_pro_advanced.py` node to fix: high-res motion loss, splice jumping, UI clutter.
- Tested: 1920x1080 maintains motion dynamics with example workflow parameters.

### 2025-01-27 (upstream)
- Fixed SVI mask dimensions from `(1,1,T,H,W)` to `(1,4,T,H,W)`.
- Added `svi_motion_strength` parameter (0.0 - 2.0, default 1.0).
- Added per-frame enable toggles (`enable_start_frame`, `enable_middle_frame`, `enable_end_frame`).

### Earlier (upstream)
- SVI Pro continuity optimization: `motion_frames` injected into latent first frame for inter-frame continuity.
- High-performance image picker node: server-side file storage, no more LocalStorage quota errors.
- Motion enhancement via `structural_repulsion_boost` (spatial gradient conditioning, high-noise stage only).
- Fixed middle-frame flickering. Recommended: high_noise_mid=0.6-0.8, low_noise_mid=0.2.

---

## Contributing

Issues and Pull Requests are welcome!

## License

See the project license file for details.
