from pathlib import Path
from typing import List, Dict, Optional
from omegaconf import DictConfig
import logging
from src.data.structures import SensorConfig, FrameData, CameraConfig
from src.data.interfaces import BaseDatasetLoader
from src.data.loaders.realsense_loader import RealSenseLoader

logger = logging.getLogger(__name__)
class DataController:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.loader: Optional[BaseDatasetLoader] = None
        self._init_loader()
        
    def _init_loader(self):
        setup = self.cfg.msalt_setup
        
        camera_configs_list = []
        if 'cameras' in setup.paths:
            for cam_yaml in setup.paths.cameras:
                intrin = Path(cam_yaml.intrinsics) if 'intrinsics' in cam_yaml else None
                extrin = Path(cam_yaml.extrinsics) if 'extrinsics' in cam_yaml else None
                
                c_cfg = CameraConfig(
                    id=cam_yaml.id,
                    name=cam_yaml.name,
                    image_path=Path(cam_yaml.image_folder),
                    intrinsics_path=intrin,
                    extrinsics_path=extrin
                )
                camera_configs_list.append(c_cfg)
        else:
            # Fallback or error if no cameras defined
            logger.warning("No 'cameras' list found in sensor_setup paths.")

        sensor_cfg = SensorConfig(
            lidar_path=Path(setup.paths.lidar_folder),
            cameras=camera_configs_list,
            ext_img=setup.extensions.images,
            ext_lidar=setup.extensions.lidar
        )
        self.loader = RealSenseLoader(sensor_cfg)
        
    def get_total_frames(self) -> int:
        return len(self.loader) if self.loader else 0
    
    def get(self, idx: int) -> FrameData:
        if not self.loader:
            raise RuntimeError("Loader not initialized")
        return self.loader.get(idx)
    
    def get_camera_ids(self):
        return self.loader.get_camera_ids() if self.loader else []
    
    def get_calibration(self, cam_id: str):
        """
        Returns {'intrinsic': np.array, 'extrinsic': np.array} 
        or None if not found.
        """
        if not self.loader:
            return None
        
        if not hasattr(self.loader, 'calibration'):
            return None
        
        calib_obj = self.loader.calibration
        if callable(calib_obj):
            calib_data = calib_obj()
        else:
            calib_data = calib_obj
            
        if not isinstance(calib_data, dict):
            return None

        return calib_data.get(cam_id)