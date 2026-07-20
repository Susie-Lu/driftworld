"""
Policy evalution: Main for computing IoU when the diffusion policy is rolled out in DriftWorld.
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/policy_eval", config_name="iou_driftworld")
def main(cfg: DictConfig):
    log.info("Main start")

    from gpc_rank.policy_eval import eval_policy_simulated
    from scipy.stats import pearsonr

    num_parallel = 100
    ep_list = [50, 100, 150, 200, 250, 300, 650]
    
    # Results
    sim_list = []
    real_list = []

    for epoch in ep_list:
        log.info(f"========== Epoch {epoch} ==========")
        avg_sim, avg_real = eval_policy_simulated(
            cfg,
            policy_ckpt=cfg.ckpt.policy_checkpoint + f"/ckpt-ep{epoch}.pth",
            epoch=epoch,
            num_parallel=num_parallel,
        )
        sim_list.append(avg_sim)
        real_list.append(avg_real)
        log.info(f"epoch {epoch} | DriftWorld IoU: {avg_sim:.4f} | real IoU: {avg_real:.4f}")

    corr, _ = pearsonr(real_list, sim_list)
    log.info(f"Pearson r^2: {corr*corr:.4f}")
    log.info("Main done")

if __name__ == "__main__":
    main()
