import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from .models import get_resnet, replace_bn_with_gn, ConditionalUnet1D

def create_policy(cfg, ckpt_path, device):
    """
    Args:
        cfg: hydra config
        ckpt_path: string of checkpoint path
        device: cpu or cuda
    Returns:
        nn.ModuleDict containing the diffusion policy ('vision_encoder' and 'invariant') with EMA weights loaded
    """
    nets = nn.ModuleDict({})

    # ResNet that extracts visual features from the raw image observations
    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets['vision_encoder'] = vision_encoder

    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2
    obs_horizon = cfg.data.obs_horizon

    # Policy: generates action sequences conditioned on the visual features and the agent's current position
    invariant = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['invariant'] = invariant 
    nets = nets.to(device)

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)
    
    # Restore checkpoint
    if not cfg.ckpt.use_official:
        ckpt = torch.load(ckpt_path, weights_only=False)
        nets.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        ema.copy_to(nets.parameters())
    else:
        for model_name, model in nets.items():
            model_state_dict = torch.load(f"{ckpt_path}/{model_name}.pth")
            model.load_state_dict(model_state_dict)

        model_state_dict = torch.load(f"{ckpt_path}/ema_nets.pth")
        ema.load_state_dict(model_state_dict)
        ema.copy_to(nets.parameters())

    return nets