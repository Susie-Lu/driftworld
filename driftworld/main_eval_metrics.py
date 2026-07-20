"""
Main for evaluating visual quality metrics for DriftWorld's rollouts on Push-T
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/train", config_name="pushT_driftworld")
def main(cfg: "DictConfig"):
    log.info("eval metrics start")
    from eval.eval_on_many_videos import evaluate_on_many_videos
    step = 1180500

    # 64-frame videos
    evaluate_on_many_videos(cfg, num_videos=1000, video_len=64, step=step)

    # Full-length videos
    evaluate_on_many_videos(cfg, num_videos=1000, video_len=None, step=step)

    log.info("eval metrics done")

if __name__ == "__main__":
    main()