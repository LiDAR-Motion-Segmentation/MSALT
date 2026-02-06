import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from benchmarking.evaluators.nuscenes_evaluator import NuScenesEvaluator

@hydra.main(config_path="../config", config_name="benchmark_nuscenes.yaml", version_base=None)
def main(cfg: DictConfig):

    # Print config to verify
    print(OmegaConf.to_yaml(cfg))

    # Run Evaluation
    evaluator = NuScenesEvaluator(cfg)
    evaluator.run()
    evaluator.print_report()    
    
if __name__ == "__main__":
    main()