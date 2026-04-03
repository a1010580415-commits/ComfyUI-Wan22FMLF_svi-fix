from comfy_api.latest import io

_WAN_I2V_OPTIONS = io.Custom("WAN_I2V_OPTIONS")

OPTIONS_DEFAULTS = {
    # Strengths
    "high_noise_start_strength": 1.0,
    "high_noise_mid_strength":   0.8,
    "low_noise_start_strength":  1.0,
    "low_noise_mid_strength":    0.2,
    "low_noise_end_strength":    1.0,
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
            ],
            outputs=[
                _WAN_I2V_OPTIONS.Output(display_name="options"),
            ],
        )

    @classmethod
    def execute(cls, options=None,
                high_noise_start_strength=1.0, high_noise_mid_strength=0.8,
                low_noise_start_strength=1.0, low_noise_mid_strength=0.2,
                low_noise_end_strength=1.0):
        out = dict(OPTIONS_DEFAULTS)
        if options:
            out.update(options)
        out.update({
            "high_noise_start_strength": high_noise_start_strength,
            "high_noise_mid_strength":   high_noise_mid_strength,
            "low_noise_start_strength":  low_noise_start_strength,
            "low_noise_mid_strength":    low_noise_mid_strength,
            "low_noise_end_strength":    low_noise_end_strength,
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
                io.Image.Input("middle_image", optional=True,
                               tooltip="Middle frame reference image for better temporal consistency"),
                io.Image.Input("end_image", optional=True,
                               tooltip="Last frame reference image (target ending)"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0,
                               step=0.01, round=0.01,
                               display_mode=io.NumberDisplay.slider, optional=True,
                               tooltip="Temporal position of middle frame (0.0 = start, 1.0 = end)"),
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
                start_image=None, middle_image=None, end_image=None,
                middle_frame_ratio=0.5,
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
        long_video_mode           = opts["long_video_mode"]
        motion_frames             = opts["motion_frames"]
        prev_latent               = opts["prev_latent"]
        continue_frames_count     = opts["continue_frames_count"]
        svi_motion_strength       = opts["svi_motion_strength"]
        video_frame_offset        = opts["video_frame_offset"]
        clip_vision_start_image   = opts["clip_vision_start_image"]
        clip_vision_middle_image  = opts["clip_vision_middle_image"]
        clip_vision_end_image     = opts["clip_vision_end_image"]

        from .wan_advanced_i2v import WanAdvancedI2V
        result = WanAdvancedI2V.execute(
            positive=positive, negative=negative, vae=vae,
            width=width, height=height, length=length, batch_size=batch_size,
            mode=mode,
            start_image=start_image, middle_image=middle_image, end_image=end_image,
            middle_frame_ratio=middle_frame_ratio,
            motion_frames=motion_frames,
            video_frame_offset=video_frame_offset,
            long_video_mode=long_video_mode,
            continue_frames_count=continue_frames_count,
            high_noise_start_strength=high_noise_start_strength,
            high_noise_mid_strength=high_noise_mid_strength,
            low_noise_start_strength=low_noise_start_strength,
            low_noise_mid_strength=low_noise_mid_strength,
            low_noise_end_strength=low_noise_end_strength,
            structural_repulsion_boost=structural_repulsion_boost,
            clip_vision_start_image=clip_vision_start_image,
            clip_vision_middle_image=clip_vision_middle_image,
            clip_vision_end_image=clip_vision_end_image,
            enable_start_frame=start_image is not None,
            enable_middle_frame=middle_image is not None,
            enable_end_frame=end_image is not None,
            svi_motion_strength=svi_motion_strength,
            prev_latent=prev_latent,
        )
        positive_high, positive_low, negative_out, latent = result.result[:4]
        return io.NodeOutput(positive_high, positive_low, negative_out, latent)
