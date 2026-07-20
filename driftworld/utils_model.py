import logging
from drifting_denoiser_multi import Denoiser

log = logging.getLogger(__name__)

def create_model(cfg, device):
    log.info("Creating DriftWorld")
    return Denoiser(
        unet_name=cfg.model.unet_name,
        temp_list=cfg.model.temp_list,
        n_neg=cfg.model.n_neg,
        num_future_frames=cfg.model.num_future_frames,
        num_history_frames=cfg.model.num_history_frames,
        decay=cfg.train.decay,
    ).to(device)