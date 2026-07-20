"""
Main for running GPC-RANK on the world model + diffusion policy
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/gpc_rank", config_name="gpc_rank_driftworld_for_ep300")
def main(cfg: DictConfig):
    log.info("Start")
    from gpc_rank.gpc_rank_eval import run_gpc_rank
    # Seed range to evaluate, [start_number_test, end_number_test).
    # To speed up the evaluation, you can split it among several GPUs and then compute the average, e.g.
    #   GPU 0:  python main_gpc_rank.py +start_number_test=0  +end_number_test=25
    #   GPU 1:  python main_gpc_rank.py +start_number_test=25 +end_number_test=50
    # Outputs go to a seed-range-specific folder, so the runs don't overwrite each other.
    start_number_test = cfg.get("start_number_test", 0)
    end_number_test = cfg.get("end_number_test", 50)

    log.info(f"Evaluating test seeds [{start_number_test}, {end_number_test})")
    run_gpc_rank(cfg, num_trial=50, num_parallel=50,
                  start_number_test=start_number_test, end_number_test=end_number_test)
    log.info("Done")

if __name__ == "__main__":
    main()
