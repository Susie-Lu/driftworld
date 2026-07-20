"""
Configurations for the action-conditioned U-Net backbone of DriftWorld
"""
from .unet_action import UNetModel

def UNet_PushT(num_history=4):
    return UNetModel(
        image_size=(96, 96),
        in_channels=(num_history + 1) * 3,
        model_channels=96,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        dropout=0,
        channel_mult=(1, 1, 1, 1),
        conv_resample=True,
        dims=3,
        action_in=2,
        action_proj=256,
        num_heads=12,
        num_head_channels=8,
        use_scale_shift_norm=True,
        resblock_updown=False,
        num_history=num_history,
    )

UNet_model_dict = {
    'UNet_PushT': UNet_PushT,
}
