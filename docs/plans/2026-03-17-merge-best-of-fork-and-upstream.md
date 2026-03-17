# Merge Best of Fork and Upstream Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Combine the fork's improvements (anchor_samples, English tooltips, sane parameter ranges) with the upstream's functional features (detail_boost actually working, motion mask decay, proper motion frame placement).

**Architecture:** Single file edit to `wan_svi_pro_advanced.py`. Take the fork as base, restore upstream's motion/mask logic, remove dead code.

**Tech Stack:** Python, PyTorch, ComfyUI node API

---

### Task 1: Remove unused import

**Files:**
- Modify: `wan_svi_pro_advanced.py:11`

**Step 1: Remove `import model_management`**

Replace line 11:
```python
import model_management
```
With nothing (delete the line).

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('wan_svi_pro_advanced.py').read()); print('OK')"`
Expected: OK

---

### Task 2: Restore detail_boost effect on frame count selection

Currently `detail_boost` is accepted as a parameter but never used. Upstream used it to scale how many frames are pulled from `prev_latent`.

**Files:**
- Modify: `wan_svi_pro_advanced.py` — the motion latent extraction block (~lines 188-196)

**Step 1: Replace the motion latent extraction block**

Current code (fork):
```python
        # --- 获取运动延续潜变量（从 prev_latent） ---
        motion_latent = None
        if prev_latent is not None:
            prev_samples = prev_latent["samples"]
            # 将 overlap_frames 转换为潜变量帧数
            motion_latent_frames = overlap_frames // 4
            if motion_latent_frames > 0:
                use_frames = min(motion_latent_frames, prev_samples.shape[2])
                if use_frames > 0:
                    motion_latent = prev_samples[:, :, -use_frames:].clone()
```

Replace with (restoring upstream's detail_boost scaling):
```python
        # --- 获取运动延续潜变量（从 prev_latent） ---
        motion_latent = None
        motion_start = 0
        motion_end = 0
        has_prev_latent = (prev_latent is not None and prev_latent.get("samples") is not None)

        if has_prev_latent:
            prev_samples = prev_latent["samples"]
            motion_latent_frames = max(1, overlap_frames // 4)
            adjusted_frames = min(motion_latent_frames, prev_samples.shape[2])

            # detail_boost scales how many frames we use from prev_latent
            if detail_boost > 1.0:
                use_frames = min(int(adjusted_frames * detail_boost), prev_samples.shape[2])
            else:
                use_frames = adjusted_frames

            motion_latent = prev_samples[:, :, -use_frames:].clone()
            motion_start = 1 if enable_start_frame else 0
            motion_end = min(motion_start + use_frames, total_latents)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('wan_svi_pro_advanced.py').read()); print('OK')"`
Expected: OK

---

### Task 3: Restore motion_boost application order (before motion_influence)

Upstream applied motion_boost first (amplify motion vectors), then motion_influence (scale overall). The fork does it in reverse order which is wrong — you'd be amplifying already-scaled vectors.

**Files:**
- Modify: `wan_svi_pro_advanced.py` — the motion_influence and motion_boost blocks (~lines 198-214)

**Step 1: Swap the order and integrate into the prev_latent block**

Remove the two separate blocks:
```python
        # --- 应用 motion_influence 缩放 ---
        if motion_latent is not None and motion_influence != 1.0:
            motion_latent = motion_latent * motion_influence

        # --- 应用 motion_boost 幅度放大（仅当有至少两帧且 boost > 1）---
        if motion_latent is not None and motion_latent.shape[2] >= 2 and motion_boost > 1.0:
            # 计算帧间差分并放大
            diffs = []
            for i in range(1, motion_latent.shape[2]):
                diff = motion_latent[:, :, i] - motion_latent[:, :, i-1]
                diffs.append(diff * motion_boost)
            # 重建
            boosted = [motion_latent[:, :, 0:1].clone()]
            for diff in diffs:
                next_frame = boosted[-1] + diff.unsqueeze(2)
                boosted.append(next_frame)
            motion_latent = torch.cat(boosted, dim=2)
```

And add to the end of the `if has_prev_latent:` block (after `motion_end = ...`):

```python
            # Apply motion_boost first (amplify frame-to-frame differences)
            if motion_latent.shape[2] >= 2 and motion_boost > 1.0:
                boosted = [motion_latent[:, :, 0:1].clone()]
                for i in range(1, motion_latent.shape[2]):
                    diff = motion_latent[:, :, i] - motion_latent[:, :, i-1]
                    boosted.append(boosted[-1] + (diff * motion_boost).unsqueeze(2))
                motion_latent = torch.cat(boosted, dim=2)

            # Then apply motion_influence (scale overall strength)
            if motion_influence != 1.0:
                motion_latent = motion_latent * motion_influence
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('wan_svi_pro_advanced.py').read()); print('OK')"`
Expected: OK

---

### Task 4: Restore upstream's image_cond_latent construction with proper motion placement

The fork concatenates anchor + motion then pads, but doesn't handle motion_start offset (skipping frame 0 when anchor is present). The upstream places motion starting at frame 1 to avoid overwriting the anchor.

**Files:**
- Modify: `wan_svi_pro_advanced.py` — the image_cond_latent construction block (~lines 216-231)

**Step 1: Replace the concatenation/padding block**

Remove:
```python
        # --- 拼接锚点与运动潜变量 ---
        if motion_latent is not None:
            image_cond_latent = torch.cat([anchor_latent, motion_latent], dim=2)
        else:
            image_cond_latent = anchor_latent

        # --- 填充到目标长度 ---
        current_frames = image_cond_latent.shape[2]
        if current_frames < total_latents:
            padding = torch.zeros(1, latent_channels, total_latents - current_frames, H, W,
                                  dtype=dtype, device=device)
            padding = comfy.latent_formats.Wan21().process_out(padding)
            image_cond_latent = torch.cat([image_cond_latent, padding], dim=2)
        elif current_frames > total_latents:
            # 截断
            image_cond_latent = image_cond_latent[:, :, :total_latents, :, :]
```

Replace with (upstream-style placement):
```python
        # --- Build image_cond_latent (upstream-style: allocate full, place anchor + motion at correct offsets) ---
        image_cond_latent = torch.zeros(1, latent_channels, total_latents, H, W,
                                        dtype=dtype, device=anchor_latent.device)
        image_cond_latent = comfy.latent_formats.Wan21().process_out(image_cond_latent)

        # Place anchor at frame 0
        if enable_start_frame and anchor_latent is not None:
            image_cond_latent[:, :, :1] = anchor_latent

        # Place motion latent at offset (after anchor if present)
        if motion_latent is not None and motion_end > motion_start:
            motion_to_use = motion_latent[:, :, :motion_end - motion_start]
            image_cond_latent[:, :, motion_start:motion_end] = motion_to_use
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('wan_svi_pro_advanced.py').read()); print('OK')"`
Expected: OK

---

### Task 5: Restore upstream's mask decay for motion frames + detail_boost decay rate

This is the critical fix. The fork leaves motion frames at mask=1 (unconditioned), losing the smooth blending that upstream provided.

**Files:**
- Modify: `wan_svi_pro_advanced.py` — the mask construction block (~lines 233-279)

**Step 1: Replace the mask block**

Remove:
```python
        # --- 构建基础掩码（单通道，0=完全条件，1=完全自由）---
        # 原版：第一帧（锚点）掩码=0，其余=1
        mask_base = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)
        mask_base[:, :, 0, :, :] = 0.0

        # --- 处理中间帧和结束帧的插入及掩码调整 ---
        # 注意：插入图像会覆盖原有内容（包括运动潜变量和填充）
        # 中间帧
        if middle_image is not None and enable_middle_frame and middle_latent_idx is not None and middle_latent_idx < total_latents:
            # 编码中间图像
            img_mid = middle_image[:1, :, :, :3]
            img_mid = comfy.utils.common_upscale(img_mid.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            mid_latent = vae.encode(img_mid)
            # 插入到指定位置（覆盖原内容）
            image_cond_latent[:, :, middle_latent_idx:middle_latent_idx+1, :, :] = mid_latent
            # 掩码将在后续根据强度调整，此处先标记，不改变 base

        # 结束帧
        if end_image is not None and enable_end_frame:
            img_end = end_image[:1, :, :, :3]
            img_end = comfy.utils.common_upscale(img_end.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            end_latent = vae.encode(img_end)
            # 插入最后一帧
            image_cond_latent[:, :, -1:, :, :] = end_latent

        # --- 生成高噪声和低噪声掩码（基于基础掩码和强度参数）---
        mask_high = mask_base.clone()
        mask_low = mask_base.clone()

        # 起始帧强度调整（锚点位置）
        if enable_start_frame:
            # 高噪声阶段
            mask_high[:, :, 0:1, :, :] = 1.0 - high_noise_start_strength
            # 低噪声阶段
            mask_low[:, :, 0:1, :, :] = 1.0 - low_noise_start_strength

        # 中间帧强度调整
        if middle_image is not None and enable_middle_frame and middle_latent_idx is not None and middle_latent_idx < total_latents:
            mask_high[:, :, middle_latent_idx:middle_latent_idx+1, :, :] = 1.0 - high_noise_mid_strength
            mask_low[:, :, middle_latent_idx:middle_latent_idx+1, :, :] = 1.0 - low_noise_mid_strength

        # 结束帧强度调整
        if end_image is not None and enable_end_frame:
            # 高噪声阶段：强制结束帧完全条件（掩码=0）
            mask_high[:, :, -1:, :, :] = 0.0
            # 低噪声阶段：根据参数
            mask_low[:, :, -1:, :, :] = 1.0 - low_noise_end_strength
```

Replace with (upstream-style with decay + image resize/encode for middle/end):
```python
        # --- Resize and encode middle/end images, insert into image_cond_latent ---
        if middle_image is not None:
            img_mid = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            img_mid = None

        if end_image is not None:
            img_end = comfy.utils.common_upscale(
                end_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        else:
            img_end = None

        # Resolve middle frame position: push past motion region if overlapping
        actual_middle_idx = middle_latent_idx if middle_latent_idx is not None else 0
        if img_mid is not None and enable_middle_frame:
            mid_latent = vae.encode(img_mid[:1, :, :, :3])
            if actual_middle_idx < total_latents:
                while actual_middle_idx < motion_end and actual_middle_idx < total_latents:
                    actual_middle_idx += 1
                if actual_middle_idx < total_latents:
                    image_cond_latent[:, :, actual_middle_idx:actual_middle_idx + 1] = mid_latent

        if img_end is not None and enable_end_frame:
            end_latent_enc = vae.encode(img_end[:1, :, :, :3])
            image_cond_latent[:, :, total_latents - 1:total_latents] = end_latent_enc

        # --- Build masks with decay for motion frames (restored from upstream) ---
        # detail_boost controls decay rate: higher = faster decay = more dynamic freedom
        if detail_boost <= 1.0:
            decay_rate = 0.9 - (detail_boost - 0.5) * 0.4
        else:
            decay_rate = 0.7 - (detail_boost - 1.0) * 0.2
        decay_rate = max(0.05, min(0.9, decay_rate))

        mask_high = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)
        mask_low = torch.ones((1, 1, total_latents, H, W), device=device, dtype=dtype)

        # Start frame strength
        if enable_start_frame and anchor_latent is not None:
            mask_high[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
            mask_low[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)

        # Motion frame decay (gradual transition from conditioned to free)
        if motion_end > motion_start:
            for i in range(motion_start, motion_end):
                distance = i - motion_start
                decay = decay_rate ** distance
                mask_high_val = 1.0 - (high_noise_start_strength * decay)
                mask_high[:, :, i:i + 1] = max(0.05, min(0.95, mask_high_val))
                mask_low_val = 1.0 - (low_noise_start_strength * decay * 0.7)
                mask_low[:, :, i:i + 1] = max(0.1, min(0.95, mask_low_val))

        # Middle frame strength
        if img_mid is not None and enable_middle_frame and actual_middle_idx < total_latents:
            mask_high[:, :, actual_middle_idx:actual_middle_idx + 1] = max(0.0, 1.0 - high_noise_mid_strength)
            mask_low[:, :, actual_middle_idx:actual_middle_idx + 1] = max(0.0, 1.0 - low_noise_mid_strength)

        # End frame strength
        if img_end is not None and enable_end_frame:
            mask_high[:, :, total_latents - 1:total_latents] = 0.0
            mask_low[:, :, total_latents - 1:total_latents] = max(0.0, 1.0 - low_noise_end_strength)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('wan_svi_pro_advanced.py').read()); print('OK')"`
Expected: OK

---

### Task 6: Update conditioning block (minor — use updated variable names)

**Files:**
- Modify: `wan_svi_pro_advanced.py` — conditioning construction block

No changes needed if variable names match. Just verify the block references `mask_high`, `mask_low`, `image_cond_latent` — which it already does.

**Step 1: Verify the full file parses and all references are consistent**

Run: `python -c "import ast; ast.parse(open('wan_svi_pro_advanced.py').read()); print('OK')"`
Expected: OK

---

### Task 7: Commit

**Step 1: Stage and commit**

```bash
git add wan_svi_pro_advanced.py
git commit -m "fix: restore detail_boost functionality and motion mask decay from upstream

- detail_boost now scales prev_latent frame count (was a dead parameter)
- Restore per-frame exponential decay on motion overlap masks for smooth blending
- Place motion latent at correct offset (after anchor frame)
- Apply motion_boost before motion_influence (correct order)
- Remove unused import model_management
- Keep fork improvements: anchor_samples, English tooltips, sane param ranges"
```
