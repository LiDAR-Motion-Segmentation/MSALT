import hydra
from omegaconf import DictConfig
from benchmarking.evaluators.semantic_kitti_evaluator import SemanticKittiEvaluator

@hydra.main(config_path="../config", config_name="benchmark_semantic_kitti", version_base=None)
def main(cfg: DictConfig):

    evaluator = SemanticKittiEvaluator(cfg)
    evaluator.run()
    evaluator.print_report()

if __name__ == "__main__":
    main()