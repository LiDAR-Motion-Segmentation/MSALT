from typing import Optional, List
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single camera sensor."""

    id: str
    name: str
    image_path: Path
    intrinsics_path: Optional[Path] = None
    extrinsics_path: Optional[Path] = None


@dataclass(frozen=True)
class SensorConfig:
    lidar_path: Path
    cameras: List[CameraConfig]
    ext_img: str = ".png"
    ext_lidar: str = ".pcd"


@dataclass
class FrameData:
    frame_index: int
    timestamp: float = 0.0
    point_cloud: Optional[np.ndarray] = None  # 3D Data: (N, 3) or (N, 4) XYZI
    images: dict = field(default_factory=dict)  # Maps 'CAM_ID' -> np.array
    metadata: dict = field(default_factory=dict)
