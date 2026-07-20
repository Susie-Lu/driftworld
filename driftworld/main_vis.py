"""
Main for visualizing (generating + saving) DriftWorld rollouts on Push-T.
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/train", config_name="pushT_driftworld")
def main(cfg: "DictConfig"):
    log.info("vis start")
    from eval.vis import visualize_videos
    step = 1180500
    fps = 10
    log.info("visualize 64-frame videos")
    visualize_videos(
        cfg,
        num_videos=32,
        video_len=64,
        step=step,
        fps=fps,
    )
    log.info("visualize full-length videos")
    visualize_videos(
        cfg,
        num_videos=32,
        video_len=None,
        step=step,
        fps=fps,
    )
    log.info("vis done")

if __name__ == "__main__":
    main()
