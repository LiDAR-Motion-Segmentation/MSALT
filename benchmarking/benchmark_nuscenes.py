import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from benchmarking.evaluators.nuscenes_evaluator import NuScenesEvaluator

@hydra.main(config_path="../config", config_name="benchmark.yaml", version_base=None)
def main(cfg: DictConfig):
    
    # # Validation
    # if Path(cfg.output_dir).exists():
    #     logging.error("You must specify output_dir! Example: python benchmark_nuscenes.py output_dir=./results")
    #     return

    # Print config to verify
    print(OmegaConf.to_yaml(cfg))

    # Run Evaluation
    evaluator = NuScenesEvaluator(cfg)
    evaluator.run()
    evaluator.print_report()    
    
if __name__ == "__main__":
    main()