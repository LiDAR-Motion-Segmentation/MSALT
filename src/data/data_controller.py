from pathlib import Path
from typing import List, Dict, Optional
from omegaconf import DictConfig

from src.data.structures import SensorConfig, FrameData
from src.data.interfaces import BaseDatasetLoader
from src.data.loaders.realsense_loader import RealSenseLoader

class DataController:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.loader: Optional[BaseDatasetLoader] = None
        self._init_loader()
        
    def _init_loader(self):
        setup = self.cfg.sensor_setup
        root = Path(setup.paths.root_dir)
        
        cam_paths = {
            cid: root / path 
            for cid, path in setup.paths.image_folders.items()
        }
        
        sensor_cfg = SensorConfig(
            root_dir=root,
            image_paths=cam_paths,
            lidar_path=root / setup.paths.lidar_folder,
            ext_img=setup.extensions.images,
            ext_lidar=setup.extensions.lidar
        )
        
        if "realsense" in setup.name.lower():
            self.loader = RealSenseLoader(sensor_cfg)
        else:
            raise ValueError(f"Unknown dataset type: {setup.name}")
        
    def get_total_frames(self) -> int:
        return len(self.loader) if self.loader else 0
    
    def get_frame(self, idx: int) -> FrameData:
        if not self.loader:
            raise RuntimeError("Loader not initialized")
        return self.loader.get_frame(idx)
    
    def get_camera_ids(self):
        return self.loader.get_camera_ids() if self.loader else []