"""
Visualize (generate + save) DriftWorld rollouts on Push-T.
"""
import os
import logging
import torch
from omegaconf import OmegaConf

from data.pushT_dataloader import get_pushT_loader_shuffleFalse, get_pushT_full_loader
from .util_eval_setup import set_seed, setup_model, save_video
from .eval_on_many_videos import _rollout_autoregressive

log = logging.getLogger(__name__)

@torch.no_grad()
def visualize_videos(cfg, num_videos=8, video_len=64, step=None, fps=2):
    """
    Args:
        cfg: Hydra config
        num_videos: number of videos to generate and save
        video_len: rollout length (overrides cfg.data.pred_horizon).
            If None, generate FULL-length videos at each episode's natural length.
        step: checkpoint step to load (None = latest)
        fps: frames per second for saved mp4s
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_mode = (video_len is None)
    log.info(f"Visualizing multiframe rollouts on Push-T (num_videos={num_videos}, "
             f"video_len={'FULL' if full_mode else video_len})")
    set_seed(cfg.train.seed)

    if not full_mode:
        OmegaConf.update(cfg, "data.pred_horizon", video_len, force_add=True)
        assert cfg.data.pred_horizon == video_len

    denoiser, device, actual_step = setup_model(cfg, step)
    n_history = denoiser.num_history_frames

    # Full-length episode vs fixed-length windows
    dataloader = get_pushT_full_loader(cfg) if full_mode else get_pushT_loader_shuffleFalse(cfg)

    folder_root = f"{cfg.output_dir}/vis"
    os.makedirs(folder_root, exist_ok=True)

    processed = 0
    for i, batch in enumerate(dataloader):
        log.info(f"(batch {i}/{len(dataloader)}) start")
        if processed >= num_videos:
            break

        # Pixels [0, 1] -> [-1, 1]
        all_obs = batch['image'].to(device)
        if cfg.data.normalize_img:
            all_obs = (all_obs - 0.5) / 0.5
        all_act = batch['action'].to(device)
        B = all_obs.shape[0]
        T = all_obs.shape[1]  # rollout length

        gt = all_obs
        gen = _rollout_autoregressive(denoiser, all_obs, all_act, n_history)  # (B, T, C, H, W) in [-1, 1]

        for j in range(B):
            if processed >= num_videos:
                break

            gen_path = f"{folder_root}/step{actual_step}_gen{processed}_ema_len{T}.mp4"
            save_video(gen[j], gen_path, fps=fps, value_range=(-1, 1))
            log.info(f"saved generated video at {gen_path}")

            gt_path = f"{folder_root}/step{actual_step}_gt{processed}_len{T}.mp4"
            save_video(gt[j], gt_path, fps=fps, value_range=(-1, 1))
            log.info(f"saved ground-truth video at {gt_path}")

            processed += 1

    log.info(f"[summary] saved {processed} generated + ground-truth videos to {folder_root}")
