"""
Utility functions for evaluation
"""
import numpy as np
import torch
import imageio
import os
import logging
import random

from utils_model import create_model
from .tensor_conversions import convert_to_uint8_np

log = logging.getLogger(__name__)

def set_seed(seed):
    log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def save_video(video_tensor, name, fps, value_range):
    """
    video_tensor: [Frames, Channels, Height, Width]
    name: filename for the video, must end in .mp4
    fps: fps for saved video
    value_range: e.g. (0, 1) or (-1, 1)
    """
    numpy_vid = convert_to_uint8_np(video_tensor, value_range)
    numpy_vid = np.transpose(numpy_vid, (0, 2, 3, 1))
    imageio.mimsave(name, numpy_vid, fps=fps)

def setup_model(cfg, step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Creating model")
    denoiser = create_model(cfg, device)
    log.info("Restoring ckpt")
    
    filepath = f"{cfg.output_dir}/ckpt_save/ckpt-latest.pth" if step==None else f"{cfg.output_dir}/ckpt_save/ckpt-step{step}.pth"
    if os.path.exists(filepath):
        ckpt = torch.load(filepath, weights_only=False)
        denoiser.load_state_dict(ckpt['model'])
        actual_step = ckpt['step']
        del ckpt
        log.info(f"Restored from step {actual_step} ckpt")
    else:
        raise ValueError(f"Checkpoint {filepath} does not exist")

    denoiser.eval()
    return denoiser, device, actual_step
