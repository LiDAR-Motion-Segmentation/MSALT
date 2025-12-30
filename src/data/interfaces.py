from abc import ABC, abstractmethod
from typing import List, Tuple
from .structures import FrameData, SensorConfig
import numpy as np

class BaseDatasetLoader(ABC):
    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self._validate_config()
        
    @abstractmethod
    def _validate_config(self) -> None:
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def get_frame(self, idx: int) -> FrameData:
        pass
    
    @abstractmethod
    def get_camera_ids(self) -> List[str]:
        pass
    
    @abstractmethod
    def calibration(self) -> np.ndarray:
        pass
