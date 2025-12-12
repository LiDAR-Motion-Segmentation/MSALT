import logging
from ntpath import exists
from tkinter import Frame
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from typing import List, Dict

from src.data.structures import FrameData, SensorConfig
from src.data.interfaces import BaseDatasetLoader

logger = logging.getLogger(__name__)

class RealSenseLoader(BaseDatasetLoader):
    def __init__(self, config: SensorConfig) -> None:
        self._lidar_files: List[Path] = []
        self._cam_files: Dict[str, List]
        super().__init__(config)
        self._index_files()
        
    def _validate_config(self) -> None:
        if not self.config.root_dir.exists():
            raise FileNotFoundError(f"Root dir missing: {self.config.root_dir}")
        logger.info(f"Initialized RealSenseLoader at {self.config.root_dir}")
        
    def _index_files(self) -> None:
        self._lidar_files = sorted(list(self.config.lidar_path.glob(f"*{self.config.ext_lidar}")))
        for cam_id, cam_path in self.config.image_paths.items():
            if cam_path.exists():
                self._cam_files[cam_id] = sorted(list(cam_path.glob(f"*{self.config.ext_img}"))) 
            else:
                logger.warning(f"Path for {cam_id} does not exist: {cam_path}")
                self._cam_files[cam_id] = []
                
    def __len__(self) -> int:
        return len(self._lidar_files)
    
    def get_camera_ids(self) -> List[str]:
        return list(self.config.image_paths.keys())
    
    def get_frame(self, idx: int) -> FrameData:
        if idx < 0 or idx >= len(self):
            raise IndexError("Frame is out of index bounds")
        
        pcd_path = str(self._lidar_files[idx])
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
        except Exception as e:
            logger.error(f"Failed to load PCD {pcd_path}: {e}")
            # points = np.zeros((0, 3))
            
        images_dict = {}
        for cam_id, files in self._cam_files.items():
            if idx < len(files):
                img = cv2.imread(str(files[idx]))
                if img is not None:
                    images_dict[cam_id] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
        return FrameData(
            frame_index=idx,
            point_cloud=points,
            images=images_dict
        )