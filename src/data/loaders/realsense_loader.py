import logging
from ntpath import exists
from tkinter import Frame
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from typing import List, Dict
from scipy.spatial.transform import Rotation as R
from pyqtgraph.console.stackwidget import exceptionChain

from src.data.structures import FrameData, SensorConfig, CameraConfig
from src.data.interfaces import BaseDatasetLoader

logger = logging.getLogger(__name__)

class RealSenseLoader(BaseDatasetLoader):
    def __init__(self, config: SensorConfig) -> None:
        self._cam_files: Dict[str, List[Path]] = {}
        self._lidar_files: List[Path] = []
        self._calibration: Dict[str, Dict] = {}
        super().__init__(config)
        self._load_calibration()
        self._index_files()
        
    def _validate_config(self) -> None:
        if not self.config.lidar_path.exists():
            raise FileNotFoundError(f"Root dir missing: {self.config.lidar_path}")
        logger.info(f"Initialized RealSenseLoader at {self.config.lidar_path}")
        
    def _index_files(self) -> None:
        self._lidar_files = sorted(list(self.config.lidar_path.glob(f"*{self.config.ext_lidar}")))
        if not self._lidar_files:
            logger.warning("No LiDAR files found! Check extension in config.")
            
        for cam_conf in self.config.cameras:
            cam_id = cam_conf.id
            folder = cam_conf.image_path
            
            if not folder.exists():
                logger.error(f"Camera folder for {cam_id} not found: {folder}")
                self._cam_files[cam_id] = []
                continue
            
            files = sorted(list(folder.glob(f"*{self.config.ext_img}")))
            self._cam_files[cam_id] = files
            
            if len(files) != len(self._lidar_files):
                logger.warning(
                    f"Sync Warning: {cam_id} has {len(files)} frames, "
                    f"LiDAR has {len(self._lidar_files)}."
                )
                
    def _load_calibration(self):
        """
        Loads .txt files as of now
        - Intrinsics: Expects 3x3 matrix.
        - Extrinsics: Expects 7 lines (x, y, z, qx, qy, qz, qw) OR (px, py, pz, qx, qy, qz, qw).
        """
        for cam_conf in self.config.cameras:
            calib_data = {'intrinsic': None, 'extrinsic': None}
            
            if cam_conf.intrinsics_path and cam_conf.intrinsics_path.exists():
                try:
                    mat = np.loadtxt(str(cam_conf.intrinsics_path))
                    calib_data['intrinsic'] = mat
                except Exception as e:
                    logger.error(f"Failed to load intrinsics for {cam_conf.id}: {e}")
                    
            if cam_conf.extrinsics_path and cam_conf.extrinsics_path.exists():
                try:
                    data = np.loadtxt(str(cam_conf.extrinsics_path))
                    if data.size == 7:
                        translation = data[:3]
                        quat = data[3:] # qx, qy, qz, qw
                        
                        # Create 4x4 Matrix
                        mat = np.eye(4)
                        mat[:3, 3] = translation
                        mat[:3, :3] = R.from_quat(quat).as_matrix()
                        
                        calib_data['extrinsic'] = mat

                    elif data.shape == (4, 4) or data.shape == (3, 4):
                        mat = data if data.shape == (4, 4) else np.vstack((data, [0,0,0,1]))
                        calib_data['extrinsic'] = mat
                        
                    else:
                        logger.warning(f"Unknown extrinsic shape {data.shape} for {cam_conf.id}")

                except Exception as e:
                    logger.error(f"Failed to load extrinsics for {cam_conf.id}: {e}")
            
            self._calibration[cam_conf.id] = calib_data
                
    def __len__(self) -> int:
        return len(self._lidar_files)
    
    def get_camera_ids(self) -> List[str]:
        return list(self._cam_files.keys())
    
    def get(self, idx: int) -> FrameData:
        if idx < 0 or idx >= len(self):
            raise IndexError("Frame is out of index bounds")
        
        points = None
        try:
            pcd_path = str(self._lidar_files[idx])
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)
        except Exception as e:
            logger.error(f"Failed to load PCD {pcd_path}: {e}")
            # points = np.zeros((0, 3))
            
        images_dict = {}
        for cam_id, files_list in self._cam_files.items():
            if idx < len(files_list):
                img = cv2.imread(str(files_list[idx]))
                if img is not None:
                    images_dict[cam_id] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
        return FrameData(
            frame_index=idx,
            point_cloud=points,
            images=images_dict,
            metadata={'calibration': self._calibration}
        )
        
    @property
    def calibration(self):
        """
        Implements the abstract property from BaseDatasetLoader.
        Returns the dictionary of camera calibrations.
        """
        return self._calibration