from pathlib import Path
from typing import List, Dict, Optional
from omegaconf import DictConfig
import logging
from src.data.structures import SensorConfig, FrameData, CameraConfig
from src.data.interfaces import BaseDatasetLoader
from src.data.loaders.realsense_loader import RealSenseLoader
from src.data.loaders.nuscenes_loader import NuScenesLoader
from src.data.loaders.kitti_loader import SemanticKittiLoader

logger = logging.getLogger(__name__)

class DataController:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.loader: Optional[BaseDatasetLoader] = None
        self._init_loader()
        
    def _init_loader(self):
        if hasattr(self.cfg, 'msalt_setup') and 'dataset_type' in self.cfg.msalt_setup:
             dtype = self.cfg.msalt_setup.dataset_type
        else:
             dtype = "custom" # Fallback to existing logic
             
        if dtype == "nuscenes":
            
            # Map Config Paths
            setup = self.cfg.msalt_setup
            root_path = Path(setup.paths.root_dir)
            version = getattr(setup, 'version', 'v1.0-mini')
            
            # Store root in 'lidar_path' slot of SensorConfig structure
            sensor_cfg = SensorConfig(
                lidar_path=root_path,
                cameras=[], # Handled internally by loader
                ext_img=setup.extensions.images,
                ext_lidar=setup.extensions.lidar,
                extra_params={
                    "version": getattr(setup, 'version', 'v1.0-mini'),
                    "sweeps": getattr(setup, 'sweeps', 10),  # Accumulate 10 sweeps for dense view
                    "scenes": getattr(setup.paths, 'scenes', [])
                }
            )
            self.loader = NuScenesLoader(sensor_cfg)
            logger.info(f"Loaded NuScenes Loader: {version}")
            
        elif dtype == "semantic_kitti":
            
            setup = self.cfg.msalt_setup
            root_path = Path(setup.paths.root_dir)
            
            # SemanticKITTI usually expects a sequence ID (e.g. "00")
            scenes = getattr(setup.paths, 'scenes', ["00"])
            
            sensor_cfg = SensorConfig(
                lidar_path=root_path,
                cameras=[],
                ext_img=".png", # KITTI defaults
                ext_lidar=".bin",
                extra_params={
                    "scenes": scenes
                }
            )
            self.loader = SemanticKittiLoader(sensor_cfg)
            logger.info(f"Loaded SemanticKITTI Loader for Sequence: {scenes[0]}")
            
        else:
            # "custom" or "realsense"
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
            logger.info("Loaded Custom/RealSense Loader")
        
    def get_total_frames(self) -> int:
        return len(self.loader) if self.loader else 0
    
    def get(self, idx: int) -> FrameData:
        if not self.loader:
            raise RuntimeError("Loader not initialized")
        return self.loader.get(idx)
    
    def get_camera_ids(self):
        
        if self.loader and hasattr(self.loader, 'get_camera_ids'):
            return self.loader.get_camera_ids()
        
        # Fallback for loaders that don't implement this explicitly yet
        if self.loader and hasattr(self.loader, 'cameras'):
             return list(self.loader.cameras.keys())
        return []
    
    def get_calibration(self, cam_id: str):
        """
        Returns {'intrinsic': np.array, 'extrinsic': np.array} 
        or None if not found.
        """
        if not self.loader:
            return None
        
        if hasattr(self.loader, 'calibration'):
            calib_obj = self.loader.calibration
            if callable(calib_obj):
                calib_data = calib_obj()
            else:
                calib_data = calib_obj
            
            if not isinstance(calib_data, dict):
                return calib_data.get(cam_id)
        
        # Fallback: Some loaders (like NuScenes) attach calib to the frame metadata.
        # We can try fetching frame 0 calibration if global is missing.    
        try:
            first_frame = self.loader.get(0)
            if first_frame and first_frame.metadata and 'calibration' in first_frame.metadata:
                  return first_frame.metadata['calibration'].get(cam_id)
        except:
            pass
        
        return None