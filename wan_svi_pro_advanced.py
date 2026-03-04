from typing_extensions import override
from comfy_api.latest import io
import torch
import node_helpers
import comfy
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
from typing import Optional
import math
import model_management


class WanSVIProAdvancedI2V(io.ComfyNode):
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent", "trim_latent", "trim_image", "next_offset")
    CATEGORY = "ComfyUI-Wan22FMLF"
    FUNCTION = "execute"
    OUTPUT_NODE = False
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WanSVIProAdvancedI2V",
            display_name="Wan SVI Pro Advanced I2V",
            category="ComfyUI-Wan22FMLF",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number,
                           tooltip="Width of the generated video in pixels"),
                io.Int.Input("height", default=480, min=16, max=8192, step=16, display_mode=io.NumberDisplay.number,
                           tooltip="Height of the generated video in pixels"),
                io.Int.Input("length", default=81, min=1, max=8192, step=4, display_mode=io.NumberDisplay.number,
                           tooltip="Total number of frames in the generated video"),
                io.Int.Input("batch_size", default=1, min=1, max=4096, display_mode=io.NumberDisplay.number,
                           tooltip="Batch size (number of videos to generate)"),
                
                # 动态调整参数（1.0 = 无增强）
                io.Float.Input("motion_boost", default=1.0, min=1.0, max=3.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Motion amplitude amplification\n1.0 = no amplification\n>1.0 = amplify movement"),
                io.Float.Input("detail_boost", default=1.0, min=1.0, max=4.0, step=0.1, round=0.1,
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Motion dynamic strength\n1.0 = balanced\n>1.0 = stronger motion dynamics"),
                io.Float.Input("motion_influence", default=1.0, min=0.0, max=2.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider,
                             tooltip="Influence strength of motion latent from previous video\n1.0 = normal, <1.0 = weaker, >1.0 = stronger"),
                
                # 重叠帧参数（以图像帧为单位，必须是4的倍数）
                io.Int.Input("overlap_frames", default=4, min=0, max=128, step=4, display_mode=io.NumberDisplay.number,
                           tooltip="Number of overlapping frames (pixel frames). 4 frames = 1 latent frame."),
                
                # 起始帧组
                io.Image.Input("start_image", optional=True,
                             tooltip="First frame reference image (anchor for the video). If provided, overrides anchor_samples."),
                io.Boolean.Input("enable_start_frame", default=True, optional=True,
                               tooltip="Enable start frame conditioning"),
                io.Float.Input("high_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in high-noise stage\n0.0 = full condition, 1.0 = no condition"),
                io.Float.Input("low_noise_start_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for start frame in low-noise stage"),
                
                # 中间帧组
                io.Image.Input("middle_image", optional=True,
                             tooltip="Middle frame reference image for better consistency"),
                io.Boolean.Input("enable_middle_frame", default=True, optional=True,
                               tooltip="Enable middle frame conditioning"),
                io.Float.Input("middle_frame_ratio", default=0.5, min=0.0, max=1.0, step=0.01, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Position of middle frame (0=start, 1=end)"),
                io.Float.Input("high_noise_mid_strength", default=0.8, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in high-noise stage"),
                io.Float.Input("low_noise_mid_strength", default=0.2, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for middle frame in low-noise stage"),
                
                # 结束帧组
                io.Image.Input("end_image", optional=True,
                             tooltip="Last frame reference image (target ending)"),
                io.Boolean.Input("enable_end_frame", default=True, optional=True,
                               tooltip="Enable end frame conditioning"),
                io.Float.Input("low_noise_end_strength", default=1.0, min=0.0, max=1.0, step=0.05, round=0.01, 
                             display_mode=io.NumberDisplay.slider, optional=True,
                             tooltip="Conditioning strength for end frame in low-noise stage"),
                
                # 其他参数
                io.ClipVisionOutput.Input("clip_vision_start_image", optional=True,
                                        tooltip="CLIP vision embedding for start frame"),
                io.ClipVisionOutput.Input("clip_vision_middle_image", optional=True,
                                        tooltip="CLIP vision embedding for middle frame"),
                io.ClipVisionOutput.Input("clip_vision_end_image", optional=True,
                                        tooltip="CLIP vision embedding for end frame"),
                
                # 原版核心输入（anchor_samples 保留，但优先使用 start_image）
                io.Latent.Input("anchor_samples", optional=True,
                              tooltip="Anchor latent samples (from VAE encode). Ignored if start_image is provided."),
                io.Latent.Input("prev_latent", optional=True,
                              tooltip="Previous video latent for seamless continuation"),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1000000, step=1, display_mode=io.NumberDisplay.number, 
                           optional=True, tooltip="Video frame offset (advanced)"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive_high"),
                io.Conditioning.Output(display_name="positive_low"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="trim_latent"),
                io.Int.Output(display_name="trim_image"),
                io.Int.Output(display_name="next_offset"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_boost=1.0, detail_boost=1.0, motion_influence=1.0,
                overlap_frames=4,
                start_image=None, enable_start_frame=True,
                high_noise_start_strength=1.0, low_noise_start_strength=1.0,
                middle_image=None, enable_middle_frame=True, middle_frame_ratio=0.5,
                high_noise_mid_strength=0.8, low_noise_mid_strength=0.2,
                end_image=None, enable_end_frame=True, low_noise_end_strength=1.0,
                clip_vision_start_image=None, clip_vision_middle_image=None,
                clip_vision_end_image=None,
                anchor_samples=None, prev_latent=None, video_frame_offset=0):
        
        # --- 基本参数计算 ---
        spatial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        total_latents = ((length - 1) // 4) + 1          # 需要的潜变量帧总数
        H = height // spatial_scale
        W = width // spatial_scale
        device = comfy.model_management.intermediate_device()

        # 空 latent（用于最终输出，采样时填充）
        latent_out = torch.zeros([batch_size, latent_channels, total_latents, H, W], device=device)

        # 偏移处理
        trim_latent = 0
        trim_image = 0
        next_offset = 0
        if video_frame_offset > 0:
            if start_image is not None and start_image.shape[0] > 1:
                start_image = start_image[video_frame_offset:] if start_image.shape[0] > video_frame_offset else None
            if middle_image is not None and middle_image.shape[0] > 1:
                middle_image = middle_image[video_frame_offset:] if middle_image.shape[0] > video_frame_offset else None
            if end_image is not None and end_image.shape[0] > 1:
                end_image = end_image[video_frame_offset:] if end_image.shape[0] > video_frame_offset else None
            next_offset = video_frame_offset + length

        # 中间帧潜变量位置（像素帧索引转潜变量帧索引）
        if middle_image is not None:
            middle_pixel_idx = int((length - 1) * middle_frame_ratio)
            middle_pixel_idx = max(4, min(middle_pixel_idx, length - 5))
            middle_latent_idx = middle_pixel_idx // 4
        else:
            middle_latent_idx = None

        # --- 获取锚点潜变量（优先使用 start_image） ---
        anchor_latent = None
        if start_image is not None and enable_start_frame:
            # 编码起始图像
            img = start_image[:1, :, :, :3]  # 取第一张，确保形状正确
            img = comfy.utils.common_upscale(img.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            anchor_latent = vae.encode(img)  # shape: [1, C, 1, H, W]
        elif anchor_samples is not None:
            anchor_latent = anchor_samples["samples"].clone()
            # 如果多帧，只取第一帧作为锚点（与后续拼接逻辑保持一致）
            if anchor_latent.shape[2] > 1:
                print("[SVI Pro Advanced] Warning: anchor_samples has multiple frames, using only the first.")
                anchor_latent = anchor_latent[:, :, :1, :, :]
        else:
            # 没有任何锚点，报错或创建零张量（这里创建零张量，但建议工作流确保有锚点）
            print("[SVI Pro Advanced] Warning: No anchor provided, using zeros.")
            anchor_latent = torch.zeros([1, latent_channels, 1, H, W], device=device)

        # 确定 dtype：使用 anchor_latent 的 dtype
        dtype = anchor_latent.dtype
        # 确保 latent_out 的 dtype 与锚点一致（虽然它只是占位，但为了统一）
        latent_out = latent_out.to(dtype)

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

        # --- 构建 conditioning ---
        positive_high = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_high
        })
        positive_low = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_low
        })
        negative_out = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": image_cond_latent,
            "concat_mask": mask_high  # 通常 negative 使用高噪声掩码
        })

        # --- 处理 CLIP vision（保持不变）---
        clip_vision_output = cls._merge_clip_vision_outputs(
            clip_vision_start_image if enable_start_frame else None,
            clip_vision_middle_image if enable_middle_frame else None,
            clip_vision_end_image if enable_end_frame else None
        )
        if clip_vision_output is not None:
            positive_low = node_helpers.conditioning_set_values(positive_low, {"clip_vision_output": clip_vision_output})
            negative_out = node_helpers.conditioning_set_values(negative_out, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent_out}

        return io.NodeOutput(positive_high, positive_low, negative_out, out_latent,
                             trim_latent, trim_image, next_offset)

    @classmethod
    def _merge_clip_vision_outputs(cls, *outputs):
        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs:
            return None
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        all_states = [o.penultimate_hidden_states for o in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result


# ===========================================
# 节点注册
# ===========================================
NODE_CLASS_MAPPINGS = {
    "WanSVIProAdvancedI2V": WanSVIProAdvancedI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSVIProAdvancedI2V": "Wan SVI Pro Advanced I2V",
}
