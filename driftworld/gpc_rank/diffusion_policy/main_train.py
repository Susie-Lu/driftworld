"""
Main for training a diffusion policy on Push-T
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="diffusion_policy")
def main(cfg: DictConfig):
    log.info("Main start")
    from train import train
    train(cfg)
    log.info("Main done")

if __name__ == "__main__":
    main()