from logging import root
from sys import meta_path
from typing import Dict, Optional, Any
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SensorConfig:
    root_dir: Path
    image_paths: Dict[str, Path]
    lidar_path: Path
    calib_path: Optional[Path] = None
    ext_img: str = ".png"
    ext_lidar: str = ".pcd"
    
@dataclass
class FrameData:
    frame_index: int
    timestamp: float = 0.0
    point_cloud: Optional[np.ndarray] = None # 3D Data: (N, 3) or (N, 4) XYZI
    
    # 2D Data: Dictionary mapping Camera ID -> Image Array (RGB)
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)