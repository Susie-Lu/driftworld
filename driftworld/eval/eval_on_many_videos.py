"""
Evaluate visual quality metrics for DriftWorld on Push-T.
"""
import time
import logging
import numpy as np
import torch
from omegaconf import OmegaConf

from data.pushT_dataloader import get_pushT_loader_shuffleFalse, get_pushT_full_loader
from .util_eval_setup import set_seed, setup_model
from .eval_metrics import get_mse, get_ssim, get_psnr, get_lpips

log = logging.getLogger(__name__)


@torch.no_grad()
def _rollout_autoregressive(denoiser, all_obs, all_act, n_history):
    """
    Autoregressive rollout: condition on the GT seed window and chain the model's own
    predictions. Returns (B, T, C, H, W), where frames 0..n_history-1 are GT and the rest predicted.
    """
    T = all_obs.shape[1]
    cur_state = all_obs[:, :n_history]  # (B, M, C, H, W) GT initial frames s_0..s_{M-1}
    actions = all_act[:, :T - 1]        # F = T-1 actions a_0..a_{T-2} -> output length T
    return denoiser.sample_autoregressive(cur_state, actions)


def _new_state():
    """Accumulators: per-video metric lists and timing totals."""
    return {
        "mse": [], "ssim": [], "psnr": [], "lpips": [],
        "total_gen_time": 0.0,
        "total_gen_frames": 0,
    }


def evaluate_on_many_videos(cfg, num_videos=1000, video_len=64, step=None):
    """
    Evaluate visual quality metrics for DriftWorld on generated videos of length video_len.

    Args:
        cfg: Hydra config
        num_videos: number of videos to evaluate
        video_len: rollout length (overrides cfg.data.pred_horizon).
            If None, generate full-length videos at each episode's natural length.
        step: checkpoint step to load (None = latest)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_mode = (video_len is None)
    log.info(f"Full-video multiframe evaluation on Push-T (num_videos={num_videos}, "
             f"video_len={'FULL' if full_mode else video_len})")
    set_seed(cfg.train.seed)

    if not full_mode:
        OmegaConf.update(cfg, "data.pred_horizon", video_len, force_add=True)
        assert cfg.data.pred_horizon == video_len

    denoiser, device, _ = setup_model(cfg, step)
    n_history = denoiser.num_history_frames
    log.info(f"num_history_steps (history frames excluded from frame metrics): {n_history}")

    # Full-length episode vs fixed-length windows
    dataloader = get_pushT_full_loader(cfg) if full_mode else get_pushT_loader_shuffleFalse(cfg)

    # Metric accumulator
    s = _new_state()

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        gen = _rollout_autoregressive(denoiser, all_obs, all_act, n_history)  # (B, T, C, H, W) in [-1, 1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_gen_time = time.perf_counter() - _t0

        batch_gen_frames = B * max(0, T - n_history)
        s["total_gen_time"] += batch_gen_time
        s["total_gen_frames"] += batch_gen_frames
        tpf = (s["total_gen_time"] / s["total_gen_frames"]) if s["total_gen_frames"] > 0 else float('nan')
        log.info(f"[timing] batch {i}: gen_time={batch_gen_time:.4f}s frames={batch_gen_frames} | "
                 f"running totals: gen_time={s['total_gen_time']:.4f}s frames={s['total_gen_frames']} "
                 f"time/frame={tpf:.4f}s")

        for j in range(B):
            if processed >= num_videos:
                break
            gt_j = gt[j]                      # (T, C, H, W)
            gt_g = gt_j[n_history:]           # generated-frame region (GT side)
            n_gen = gt_g.shape[0]
            if n_gen <= 0:
                log.warning(f"video {processed} (batch {i} sample {j}) has length {T} <= "
                            f"history {n_history}; skipping metrics")
                processed += 1
                continue

            gen_j = gen[j]                    # (T, C, H, W)
            gen_g = gen_j[n_history:]

            mse_m = float(get_mse(gen_g, gt_g).mean())
            ssim_m = float(get_ssim(gen_g, gt_g).mean())
            psnr_m = float(get_psnr(gen_g, gt_g).mean())
            lpips_m = float(get_lpips(gen_g, gt_g).mean())

            s["mse"].append(mse_m); s["ssim"].append(ssim_m); s["psnr"].append(psnr_m)
            s["lpips"].append(lpips_m)

            log.info(f"[metrics] video {processed} (batch {i} sample {j}, "
                     f"len={T}, n_gen={n_gen}): MSE={mse_m:.5f} SSIM={ssim_m:.5f} "
                     f"PSNR={psnr_m:.5f} LPIPS={lpips_m:.5f}")

            log.info(f"[running avg] over {len(s['mse'])} videos: "
                     f"MSE={np.mean(s['mse']):.5f} SSIM={np.mean(s['ssim']):.5f} "
                     f"PSNR={np.mean(s['psnr']):.5f} LPIPS={np.mean(s['lpips']):.5f}")

            processed += 1

    tpf = (s["total_gen_time"] / s["total_gen_frames"]) if s["total_gen_frames"] > 0 else float('nan')
    log.info(f"[timing] FINAL: total gen_time={s['total_gen_time']:.3f}s over "
             f"{s['total_gen_frames']} generated frames | time/frame={tpf:.5f}s")

    if len(s["mse"]) == 0:
        log.info("[summary] no videos were evaluated")
        return

    log.info(
        f"[summary] per-video averages over {len(s['mse'])} videos: "
        f"MSE={np.mean(s['mse']):.5f} SSIM={np.mean(s['ssim']):.5f} PSNR={np.mean(s['psnr']):.5f} "
        f"LPIPS={np.mean(s['lpips']):.5f}"
    )
