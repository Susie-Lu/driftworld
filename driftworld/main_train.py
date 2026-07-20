"""
Main for training DriftWorld
torchrun --nproc_per_node=2 main_train.py --config-name=pushT_driftworld
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/train", config_name="pushT_driftworld")
def main(cfg: DictConfig):
    log.info("Main start")
    from train import train
    train(cfg)
    log.info("Main done")

if __name__ == "__main__":
    main()