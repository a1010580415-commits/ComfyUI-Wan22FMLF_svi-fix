from comfy_api.latest import io
import torch
import node_helpers
import comfy
import comfy.utils
import comfy.latent_formats
from .utils import merge_clip_vision_outputs, apply_repulsion_boost

_WAN_I2V_OPTIONS = io.Custom("WAN_I2V_OPTIONS")

OPTIONS_DEFAULTS = {
    # Strengths
    "high_noise_start_strength": 1.0,
    "high_noise_mid_strength":   0.8,
    "low_noise_start_strength":  1.0,
    "low_noise_mid_strength":    0.2,
    "low_noise_end_strength":    1.0,
    "middle_frame_ratio":        0.5,
    # Continuation
    "long_video_mode":           "DISABLED",
    "motion_frames":             None,
    "prev_latent":               None,
    "continue_frames_count":     5,
    "svi_motion_strength":       1.0,
    "video_frame_offset":        0,
    # Clip vision
    "clip_vision_start_image":   None,
    "clip_vision_middle_image":  None,
    "clip_vision_end_image":     None,
}


class WanStrengthOptions(io.ComfyNode):
    RETURN_TYPES = ("WAN_I2V_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanStrengthOptions",
            display_name="Wan Strength Options",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                _WAN_I2V_OPTIONS.Input("options", optional=True,
                                       tooltip="Chain from another options node"),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Conditioning strength for start frame in high-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Conditioning strength for middle frame in high-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Conditioning strength for start frame in low-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Conditioning strength for middle frame in low-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Conditioning strength for end frame in low-noise stage\n0.0 = ignore, 1.0 = fully conditioned"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0,
                               step=0.01, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Temporal position of middle frame (0.0 = start, 1.0 = end)"),
            ],
            outputs=[
                _WAN_I2V_OPTIONS.Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None,
                high_noise_start_strength=1.0, high_noise_mid_strength=0.8,
                low_noise_start_strength=1.0, low_noise_mid_strength=0.2,
                low_noise_end_strength=1.0, middle_frame_ratio=0.5):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "high_noise_start_strength": high_noise_start_strength,
            "high_noise_mid_strength":   high_noise_mid_strength,
            "low_noise_start_strength":  low_noise_start_strength,
            "low_noise_mid_strength":    low_noise_mid_strength,
            "low_noise_end_strength":    low_noise_end_strength,
            "middle_frame_ratio":        middle_frame_ratio,
        })
        return io.NodeOutput(out)


class WanContinuationOptions(io.ComfyNode):
    RETURN_TYPES = ("WAN_I2V_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanContinuationOptions",
            display_name="Wan Continuation Options",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                _WAN_I2V_OPTIONS.Input("options", optional=True,
                                       tooltip="Chain from another options node"),
                io.Combo.Input("long_video_mode",
                               ["AUTO_CONTINUE", "SVI", "LATENT_CONTINUE"],
                               default="SVI", optional=True,
                               tooltip="AUTO_CONTINUE = pixel-space continuation via motion_frames\nSVI = latent-space continuation via prev_latent\nLATENT_CONTINUE = direct latent injection at frame 0"),
                io.Image.Input("motion_frames", optional=True,
                               tooltip="Last N frames from previous video (AUTO_CONTINUE mode)"),
                io.Latent.Input("prev_latent", optional=True,
                                tooltip="Previous video latent (SVI / LATENT_CONTINUE mode)"),
                io.Int.Input("continue_frames_count", default=5, min=0, max=20, step=1,
                             display_mode=io.NumberDisplay.number, optional=True,
                             tooltip="Number of frames to use from previous video for continuation"),
                io.Float.Input("svi_motion_strength", default=1.0, min=0.0, max=2.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="SVI mode motion intensity\n<1.0 = more stable, >1.0 = more dynamic"),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1,
                             display_mode=io.NumberDisplay.number, optional=True,
                             tooltip="Frame offset for image sequences spanning multiple clips"),
            ],
            outputs=[
                _WAN_I2V_OPTIONS.Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None, long_video_mode="SVI",
                motion_frames=None, prev_latent=None,
                continue_frames_count=5, svi_motion_strength=1.0,
                video_frame_offset=0):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "long_video_mode":       long_video_mode,
            "motion_frames":         motion_frames,
            "prev_latent":           prev_latent,
            "continue_frames_count": continue_frames_count,
            "svi_motion_strength":   svi_motion_strength,
            "video_frame_offset":    video_frame_offset,
        })
        return io.NodeOutput(out)


class WanClipVisionOptions(io.ComfyNode):
    RETURN_TYPES = ("WAN_I2V_OPTIONS",)
    RETURN_NAMES = ("options",)
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanClipVisionOptions",
            display_name="Wan Clip Vision Options",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                _WAN_I2V_OPTIONS.Input("options", optional=True,
                                       tooltip="Chain from another options node"),
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True,
                                          tooltip="CLIP vision embedding for start frame (Wan 2.1 FLF only)"),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True,
                                          tooltip="CLIP vision embedding for middle frame (Wan 2.1 FLF only)"),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True,
                                          tooltip="CLIP vision embedding for end frame (Wan 2.1 FLF only)"),
            ],
            outputs=[
                _WAN_I2V_OPTIONS.Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None,
                clip_vision_start_image=None,
                clip_vision_middle_image=None,
                clip_vision_end_image=None):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "clip_vision_start_image":  clip_vision_start_image,
            "clip_vision_middle_image": clip_vision_middle_image,
            "clip_vision_end_image":    clip_vision_end_image,
        })
        return io.NodeOutput(out)


class WanI2VBase(io.ComfyNode):
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanI2VBase",
            display_name="Wan I2V",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                # Core
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16,
                             display_mode=io.NumberDisplay.number,
                             tooltip="Width of the generated video in pixels"),
                io.Int.Input("height", default=480, min=16, max=8192, step=16,
                             display_mode=io.NumberDisplay.number,
                             tooltip="Height of the generated video in pixels"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4,
                             display_mode=io.NumberDisplay.number,
                             tooltip="Total number of frames in the generated video"),
                io.Int.Input("batch_size", default=1, min=1, max=4096,
                             display_mode=io.NumberDisplay.number,
                             tooltip="Number of videos to generate"),
                # Frame images
                io.Image.Input("start_image", optional=True,
                               tooltip="First frame reference image (anchor)"),
                io.Boolean.Input("enable_start_frame", default=True, optional=True,
                                 tooltip="Enable start frame conditioning"),
                io.Image.Input("middle_image", optional=True,
                               tooltip="Middle frame reference image for better temporal consistency"),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True,
                                 tooltip="Enable middle frame conditioning"),
                io.Image.Input("end_image", optional=True,
                               tooltip="Last frame reference image (target ending)"),
                io.Boolean.Input("enable_end_frame", default=True, optional=True,
                                 tooltip="Enable end frame conditioning"),
                # Mode
                io.Combo.Input("mode", ["NORMAL", "SINGLE_PERSON"], default="NORMAL", optional=True,
                               tooltip="NORMAL = all frames condition both stages\nSINGLE_PERSON = only start frame conditions low-noise stage"),
                io.Float.Input("structural_repulsion_boost", default=1.0, min=1.0, max=2.0,
                               step=0.05, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Enhances motion between reference frames using spatial gradients\n1.0 = disabled, >1.0 = stronger repulsion in high-motion areas\nOnly affects high-noise stage"),
                # Options chain
                _WAN_I2V_OPTIONS.Input("options", optional=True,
                                       tooltip="Connect Wan Strength Options, Wan Continuation Options, or Wan Clip Vision Options"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                start_image=None, enable_start_frame=True,
                middle_image=None, enable_middle_frame=True,
                end_image=None, enable_end_frame=True,
                mode="NORMAL", structural_repulsion_boost=1.0,
                options=None):

        opts = dict(OPTIONS_DEFAULTS)
        if options:
            opts.update(options)

        high_noise_start_strength = opts["high_noise_start_strength"]
        high_noise_mid_strength   = opts["high_noise_mid_strength"]
        low_noise_start_strength  = opts["low_noise_start_strength"]
        low_noise_mid_strength    = opts["low_noise_mid_strength"]
        low_noise_end_strength    = opts["low_noise_end_strength"]
        middle_frame_ratio        = opts["middle_frame_ratio"]
        long_video_mode           = opts["long_video_mode"]
        motion_frames             = opts["motion_frames"]
        prev_latent               = opts["prev_latent"]
        continue_frames_count     = opts["continue_frames_count"]
        svi_motion_strength       = opts["svi_motion_strength"]
        video_frame_offset        = opts["video_frame_offset"]
        clip_vision_start_image   = opts["clip_vision_start_image"]
        clip_vision_middle_image  = opts["clip_vision_middle_image"]
        clip_vision_end_image     = opts["clip_vision_end_image"]

        enable_start_frame  = enable_start_frame  and start_image  is not None
        enable_middle_frame = enable_middle_frame and middle_image is not None
        enable_end_frame    = enable_end_frame    and end_image    is not None

        spacial_scale   = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t        = ((length - 1) // 4) + 1
        device          = comfy.model_management.intermediate_device()

        latent = torch.zeros([batch_size, latent_channels, latent_t,
                              height // spacial_scale, width // spacial_scale],
                             device=device)

        has_motion_frames  = (motion_frames is not None and motion_frames.shape[0] > 0)
        is_pure_triple_mode = (not has_motion_frames and long_video_mode == "DISABLED")

        if video_frame_offset >= 0:
            if (long_video_mode in ("AUTO_CONTINUE", "SVI")) and has_motion_frames and continue_frames_count > 0:
                actual_count  = min(continue_frames_count, motion_frames.shape[0])
                motion_frames = motion_frames[-actual_count:]
                video_frame_offset = max(0, video_frame_offset - motion_frames.shape[0])

            if video_frame_offset > 0:
                if start_image is not None and start_image.shape[0] > 1:
                    start_image = start_image[video_frame_offset:] if start_image.shape[0] > video_frame_offset else None
                if middle_image is not None and middle_image.shape[0] > 1:
                    middle_image = middle_image[video_frame_offset:] if middle_image.shape[0] > video_frame_offset else None
                if end_image is not None and end_image.shape[0] > 1:
                    end_image = end_image[video_frame_offset:] if end_image.shape[0] > video_frame_offset else None

        if motion_frames is not None:
            motion_frames = comfy.utils.common_upscale(
                motion_frames.movedim(-1, 1), width, height, "area", "center"
            ).movedim(1, -1)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length if is_pure_triple_mode else 1].movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)

        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length if is_pure_triple_mode else -1:].movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)

        image     = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones((1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]), device=device)

        desired_pixel_idx = int(length * middle_frame_ratio)
        middle_idx = (desired_pixel_idx // 4) * 4
        middle_idx = max(4, min(middle_idx, length - 5))

        mask_high_noise = mask_base.clone()
        mask_low_noise  = mask_base.clone()

        svi_continue_mode = False

        # --- LATENT_CONTINUE ---
        latent_continue_mode   = False
        prev_latent_for_concat = None
        if long_video_mode == "LATENT_CONTINUE":
            has_prev = prev_latent is not None and prev_latent.get("samples") is not None
            if has_prev and continue_frames_count > 0 and start_image is None:
                latent_continue_mode = True
                prev_samples = prev_latent["samples"]
                if prev_samples.shape[2] > 0:
                    last_frame = prev_samples[:, :, -1:].clone()
                    lH, lW = latent.shape[-2], latent.shape[-1]
                    if last_frame.shape[-2] != lH or last_frame.shape[-1] != lW:
                        last_frame = torch.nn.functional.interpolate(
                            last_frame.squeeze(2), size=(lH, lW), mode="bilinear", align_corners=False
                        ).unsqueeze(2)
                    for b in range(batch_size):
                        latent[b:b+1, :, 0:1] = last_frame
                    mask_high_noise[:, :, :4] = 0.0
                    mask_low_noise[:, :, :4]  = 0.0
                    prev_latent_for_concat = last_frame

        # --- SVI ---
        if long_video_mode == "SVI":
            total_latents    = latent_t
            H                = height // spacial_scale
            W                = width  // spacial_scale
            middle_latent_idx = middle_idx // 4
            end_latent_idx   = total_latents - 1
            has_prev = prev_latent is not None and prev_latent.get("samples") is not None

            def _build_svi_conditioning(anchor_latent, motion_lat=None):
                image_cond = torch.zeros(1, latent_channels, total_latents, H, W,
                                         dtype=anchor_latent.dtype, device=anchor_latent.device)
                image_cond = comfy.latent_formats.Wan21().process_out(image_cond)
                if enable_start_frame:
                    image_cond[:, :, :1] = anchor_latent
                if motion_lat is not None:
                    m_end = min(1 + motion_lat.shape[2], total_latents)
                    image_cond[:, :, 1:m_end] = motion_lat[:, :, :m_end - 1]
                if middle_image is not None and enable_middle_frame:
                    ml = vae.encode(middle_image[:1, :, :, :3])
                    if middle_latent_idx < total_latents:
                        image_cond[:, :, middle_latent_idx:middle_latent_idx + 1] = ml
                if end_image is not None and enable_end_frame:
                    el = vae.encode(end_image[:1, :, :, :3])
                    image_cond[:, :, end_latent_idx:end_latent_idx + 1] = el

                mask_h = torch.ones((1, 4, total_latents, H, W), dtype=anchor_latent.dtype, device=device)
                mask_l = torch.ones((1, 4, total_latents, H, W), dtype=anchor_latent.dtype, device=device)
                if enable_start_frame:
                    mask_h[:, :, :1] = max(0.0, 1.0 - high_noise_start_strength)
                    mask_l[:, :, :1] = max(0.0, 1.0 - low_noise_start_strength)
                if middle_image is not None and enable_middle_frame:
                    sr = max(0, middle_latent_idx)
                    er = min(total_latents, middle_latent_idx + 1)
                    mask_h[:, :, sr:er] = max(0.0, 1.0 - high_noise_mid_strength)
                    mask_l[:, :, sr:er] = max(0.0, 1.0 - low_noise_mid_strength)
                if end_image is not None and enable_end_frame:
                    mask_h[:, :, end_latent_idx:end_latent_idx + 1] = 0.0
                    mask_l[:, :, end_latent_idx:end_latent_idx + 1] = max(0.0, 1.0 - low_noise_end_strength)
                return image_cond, mask_h, mask_l

            if has_prev and continue_frames_count > 0:
                svi_continue_mode = True
                if start_image is not None:
                    anchor = vae.encode(start_image[:1, :, :, :3])
                else:
                    anchor = torch.zeros([1, latent_channels, 1, H, W], device=device, dtype=latent.dtype)
                prev_samples = prev_latent["samples"]
                n_motion = min(prev_samples.shape[2], ((continue_frames_count - 1) // 4) + 1)
                motion_lat = prev_samples[:, :, -n_motion:].clone()
                if svi_motion_strength != 1.0:
                    motion_lat = motion_lat * svi_motion_strength
                image_cond, mask_h, mask_l = _build_svi_conditioning(anchor, motion_lat)
            elif start_image is not None:
                anchor = vae.encode(start_image[:1, :, :, :3])
                image_cond, mask_h, mask_l = _build_svi_conditioning(anchor)
            else:
                image_cond = comfy.latent_formats.Wan21().process_out(
                    torch.zeros(1, latent_channels, total_latents, H, W, device=device, dtype=latent.dtype)
                )
                mask_h = torch.ones((1, 4, total_latents, H, W), device=device, dtype=latent.dtype)
                mask_l = torch.ones((1, 4, total_latents, H, W), device=device, dtype=latent.dtype)

            pos_high = node_helpers.conditioning_set_values(positive, {"concat_latent_image": image_cond, "concat_mask": mask_h})
            pos_low  = node_helpers.conditioning_set_values(positive, {"concat_latent_image": image_cond, "concat_mask": mask_l})
            neg_out  = node_helpers.conditioning_set_values(negative, {"concat_latent_image": image_cond, "concat_mask": mask_h})

            clip_vision_output = merge_clip_vision_outputs(
                clip_vision_start_image  if enable_start_frame  else None,
                clip_vision_middle_image if enable_middle_frame else None,
                clip_vision_end_image    if enable_end_frame    else None,
            )
            if clip_vision_output is not None:
                pos_low = node_helpers.conditioning_set_values(pos_low, {"clip_vision_output": clip_vision_output})
                neg_out = node_helpers.conditioning_set_values(neg_out, {"clip_vision_output": clip_vision_output})

            return io.NodeOutput(pos_high, pos_low, neg_out, {"samples": latent})

        # --- AUTO_CONTINUE / DISABLED / LATENT_CONTINUE (pixel-space path) ---
        if has_motion_frames and long_video_mode not in ("SVI", "LATENT_CONTINUE"):
            image[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            motion_latent_frames = ((motion_frames.shape[0] - 1) // 4) + 1
            mask_high_noise[:, :, :motion_latent_frames * 4] = 0.0
            if not svi_continue_mode:
                mask_low_noise[:, :, :motion_latent_frames * 4] = 0.0
            if middle_image is not None and enable_middle_frame:
                image[middle_idx:middle_idx + 1] = middle_image
                sr, er = max(0, middle_idx), min(length, middle_idx + 4)
                mask_high_noise[:, :, sr:er] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low_noise[:, :, sr:er]  = max(0.0, 1.0 - low_noise_mid_strength)
            if end_image is not None and enable_end_frame:
                image[-1:] = end_image[:, :, :, :3]
                mask_high_noise[:, :, -1:] = 0.0
                mask_low_noise[:, :, -1:]  = max(0.0, 1.0 - low_noise_end_strength)
        else:
            if start_image is not None and long_video_mode != "LATENT_CONTINUE" and enable_start_frame:
                image[:start_image.shape[0]] = start_image[:, :, :, :3]
                if is_pure_triple_mode:
                    mask_range = min(start_image.shape[0] + 3, length)
                    mask_high_noise[:, :, :mask_range] = max(0.0, 1.0 - high_noise_start_strength)
                    mask_low_noise[:, :, :mask_range]  = max(0.0, 1.0 - low_noise_start_strength)
                else:
                    slf = ((start_image.shape[0] - 1) // 4) + 1
                    mask_high_noise[:, :, :slf * 4] = max(0.0, 1.0 - high_noise_start_strength)
                    mask_low_noise[:, :, :slf * 4]  = max(0.0, 1.0 - low_noise_start_strength)
            if middle_image is not None and enable_middle_frame:
                image[middle_idx:middle_idx + 1] = middle_image
                sr, er = max(0, middle_idx), min(length, middle_idx + 4)
                mask_high_noise[:, :, sr:er] = max(0.0, 1.0 - high_noise_mid_strength)
                mask_low_noise[:, :, sr:er]  = max(0.0, 1.0 - low_noise_mid_strength)
            if end_image is not None and enable_end_frame:
                image[-end_image.shape[0]:] = end_image[:, :, :, :3]
                if is_pure_triple_mode:
                    mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
                    mask_low_noise[:, :, -end_image.shape[0]:]  = max(0.0, 1.0 - low_noise_end_strength)
                else:
                    mask_high_noise[:, :, -1:] = 0.0
                    mask_low_noise[:, :, -1:]  = max(0.0, 1.0 - low_noise_end_strength)

        if latent_continue_mode and prev_latent_for_concat is not None:
            concat_latent_image = vae.encode(image[:, :, :, :3])
            concat_latent_image[:, :, 0:1] = prev_latent_for_concat
        else:
            concat_latent_image = vae.encode(image[:, :, :, :3])

        if structural_repulsion_boost > 1.001 and length > 4:
            ref_indices = []
            if start_image is not None and enable_start_frame and long_video_mode != "LATENT_CONTINUE":
                ref_indices.append(0)
            if middle_image is not None and enable_middle_frame:
                ref_indices.append(middle_idx // 4)
            if end_image is not None and enable_end_frame:
                ref_indices.append(latent_t - 1)
            if len(ref_indices) >= 2:
                concat_latent_image = apply_repulsion_boost(concat_latent_image, ref_indices, structural_repulsion_boost)

        if latent_continue_mode:
            concat_latent_image_low = concat_latent_image
        elif mode == "SINGLE_PERSON":
            image_low = torch.ones((length, height, width, 3), device=device) * 0.5
            if motion_frames is not None:
                image_low[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            elif start_image is not None:
                image_low[:start_image.shape[0]] = start_image[:, :, :, :3]
            concat_latent_image_low = vae.encode(image_low[:, :, :, :3])
        elif low_noise_start_strength == 0.0 or low_noise_mid_strength == 0.0 or low_noise_end_strength == 0.0:
            image_low = torch.ones((length, height, width, 3), device=device) * 0.5
            if motion_frames is not None and low_noise_start_strength > 0.0:
                image_low[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
            elif start_image is not None and low_noise_start_strength > 0.0 and enable_start_frame:
                image_low[:start_image.shape[0]] = start_image[:, :, :, :3]
            if middle_image is not None and low_noise_mid_strength > 0.0 and enable_middle_frame:
                image_low[middle_idx:middle_idx + 1] = middle_image
            if end_image is not None and low_noise_end_strength > 0.0 and enable_end_frame:
                if is_pure_triple_mode:
                    image_low[-end_image.shape[0]:] = end_image[:, :, :, :3]
                else:
                    image_low[-1:] = end_image[:, :, :, :3]
            concat_latent_image_low = vae.encode(image_low[:, :, :, :3])
        else:
            concat_latent_image_low = concat_latent_image

        mask_high_reshaped = mask_high_noise.view(
            1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]
        ).transpose(1, 2)
        mask_low_reshaped = mask_low_noise.view(
            1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]
        ).transpose(1, 2)

        pos_high = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image,     "concat_mask": mask_high_reshaped})
        pos_low  = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image_low, "concat_mask": mask_low_reshaped})
        neg_out  = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image,     "concat_mask": mask_high_reshaped})

        clip_vision_output = merge_clip_vision_outputs(
            clip_vision_start_image  if enable_start_frame  else None,
            clip_vision_middle_image if enable_middle_frame else None,
            clip_vision_end_image    if enable_end_frame    else None,
        )
        if clip_vision_output is not None:
            pos_low = node_helpers.conditioning_set_values(pos_low, {"clip_vision_output": clip_vision_output})
            neg_out = node_helpers.conditioning_set_values(neg_out, {"clip_vision_output": clip_vision_output})

        return io.NodeOutput(pos_high, pos_low, neg_out, {"samples": latent})
