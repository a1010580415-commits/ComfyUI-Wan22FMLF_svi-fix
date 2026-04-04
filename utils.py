import torch
import torch.nn.functional as F
import comfy.clip_vision


def merge_clip_vision_outputs(*outputs):
    """Merge multiple CLIP vision outputs by concatenating hidden states."""
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


def apply_repulsion_boost(concat_latent, ref_latent_indices, boost):
    """Enhance motion between reference frames in concat_latent_image.

    Implements PainterFLF2V's "Anti-Ghost Vector" technique:
    for each transition frame between two reference latents, amplifies the
    high-frequency difference from a linear interpolation between the references.
    This pushes the model away from flat/ghosted linear morph paths.

    concat_latent: [1, C, T, H, W]
    ref_latent_indices: sorted list of latent frame indices with real content
    boost: float > 1.0  (1.0 = no effect)
    """
    if boost <= 1.001 or len(ref_latent_indices) < 2:
        return concat_latent

    scale = (boost - 1.0) * 4.0
    result = concat_latent.clone()

    for i in range(len(ref_latent_indices) - 1):
        t1 = ref_latent_indices[i]
        t2 = ref_latent_indices[i + 1]
        if t2 - t1 < 2:
            continue

        ref1 = concat_latent[:, :, t1:t1 + 1]  # [1, C, 1, H, W]
        ref2 = concat_latent[:, :, t2:t2 + 1]

        for t in range(t1 + 1, t2):
            alpha = (t - t1) / (t2 - t1)
            linear = (1.0 - alpha) * ref1 + alpha * ref2
            diff = concat_latent[:, :, t:t + 1] - linear  # [1, C, 1, H, W]

            # Spatial high-pass: remove low-frequency color drift, keep structure
            diff_2d = diff.squeeze(2)  # [1, C, H, W]
            low_freq = F.avg_pool2d(diff_2d, kernel_size=3, stride=1, padding=1)
            high_freq = (diff_2d - low_freq).unsqueeze(2)  # [1, C, 1, H, W]

            result[:, :, t:t + 1] = concat_latent[:, :, t:t + 1] + high_freq * scale

    return result
