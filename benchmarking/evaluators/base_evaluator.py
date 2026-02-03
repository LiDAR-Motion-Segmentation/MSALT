from abc import ABC, abstractmethod
from omegaconf import DictConfig

class BaseEvaluator(ABC):
    """
    Abstract Base Class for all dataset evaluators.
    Now configured via Hydra DictConfig.
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def print_report(self):
        pass