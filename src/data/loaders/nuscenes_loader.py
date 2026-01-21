import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import cv2

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
except ImportError:
    raise ImportError("Please run 'pip install nuscenes-devkit' to use this loader.")

from src.data.structures import FrameData, SensorConfig
from src.data.interfaces import BaseDatasetLoader
from src.core.objects import BoundingBox3D

logger = logging.getLogger(__name__)

class NuScenesLoader(BaseDatasetLoader):
    def __init__(self, config: SensorConfig) -> None:
        self.root = config.lidar_path
        self.version = config.extra_params.get("version", "v1.0-mini")
        self.default_sweeps = config.extra_params.get("sweeps", 1)
        self.scenes_filter = config.extra_params.get('scenes', [])
        
        logger.info(f"Initializing NuScenes ({self.version}) at {self.root}...")
        self.nusc = NuScenes(version=self.version, dataroot=str(self.root), verbose=False)
        
        if not self.scenes_filter:
            # Load everything (Default)
            self.samples = self.nusc.sample
        else:
            self.samples = []
            logger.info(f"Filtering for scenes: {self.scenes_filter}")
            
            for req in self.scenes_filter:
                scene_rec = None
                
                # Case A: Integer Index (e.g., Scene 0)
                if isinstance(req, int):
                    if 0 <= req < len(self.nusc.scene):
                        scene_rec = self.nusc.scene[req]
                    else:
                        logger.warning(f"Scene index {req} out of bounds.")
                
                # Case B: String Name (e.g., "scene-0061")
                elif isinstance(req, str):
                    # Search by name (slower but safer)
                    for s in self.nusc.scene:
                        if s['name'] == req:
                            scene_rec = s
                            break
                    if not scene_rec:
                         logger.warning(f"Scene '{req}' not found.")
                
                # If we found the scene, traverse its samples
                if scene_rec:
                    current_token = scene_rec['first_sample_token']
                    while current_token:
                        sample = self.nusc.get('sample', current_token)
                        self.samples.append(sample)
                        current_token = sample['next']
                        
        logger.info(f"Loaded {len(self.samples)} frames.")
        
        # self.samples = self.nusc.sample
        self._camera_channels = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
                
        super().__init__(config)
        
    def _validate_config(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"NuScenes root not found: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_camera_ids(self) -> List[str]:
        return self._camera_channels
    
    def get(self, idx: int) -> FrameData:
        sample = self.samples[idx]
        lidar_token = sample['data']['LIDAR_TOP']
        
        # 1. LOAD & TRANSFORM POINTS (Sensor -> Ego)
        # This fixes the "Axis Problem". We visualize in EGO frame.
        # Note: Using single sweep to avoid Y-axis overlap from ego pose transformations in multisweep
        # If you need dense clouds, consider implementing point deduplication or filtering
        points, T_sensor_to_ego = self._get_single_sweep_strict(lidar_token)
        
        #  LOAD GROUND TRUTH 
        gt_boxes = []
        raw_boxes = self.nusc.get_boxes(lidar_token)
        
        for box in raw_boxes:
            # box is a NuScenes Box object in SENSOR frame.
            # We must transform it to EGO frame to match the point cloud.
            
            # Apply Rotation/Translation
            # T_sensor_to_ego is (4,4) matrix
            # Box center:
            # center = np.array([box.center[0], box.center[1], box.center[2], 1.0])
            # center_ego = T_sensor_to_ego @ center
            
            # # Box Orientation:
            # calib = self.nusc.get('calibrated_sensor', self.nusc.get('sample_data', lidar_token)['calibrated_sensor_token'])
            # rot_quat = Quaternion(calib['rotation'])
            # new_orientation = rot_quat * box.orientation
            
            # Calculate Yaw (Heading) from Quaternion
            # Nuscenes Yaw is usually around Z-axis
            v = np.dot(box.orientation.rotation_matrix, np.array([1, 0, 0]))
            heading = np.arctan2(v[1], v[0])
            
            # Create YOUR BoundingBox3D
            b = BoundingBox3D(
                x=box.center[0], 
                y=box.center[1], 
                z=box.center[2],
                dx=box.wlh[1], # NuScenes is w, l, h -> dx(len), dy(wid), dz(hgt)
                dy=box.wlh[0], # Map L->dx, W->dy carefully. NuScenes often swaps these relative to Kitti.
                dz=box.wlh[2],
                heading=heading,
                label=box.name, # e.g. 'vehicle.car'
                confidence=1.0
            )
            gt_boxes.append(b)

        # 3. CAMERAS & CALIBRATION (Ego Frame)
        images_dict = {}
        calibration_dict = {}
        
        # Get LiDAR->Ego transform for this frame (should be Identity if we transformed points?)
        # Actually, since we transformed points to Ego, the "Extrinsics" we pass to the visualizer
        # should represent Camera->Ego.
        
        for cam_channel in self._camera_channels:
            if cam_channel in sample['data']:
                cam_token = sample['data'][cam_channel]
                cam_data = self.nusc.get('sample_data', cam_token)
            
                img_path = Path(self.root) / cam_data['filename']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    images_dict[cam_channel] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                    # We need: World(Ego) -> Camera
                    cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
                    # Camera -> Ego
                    T_cam_to_ego = transform_matrix(
                        cam_calib['translation'], 
                        Quaternion(cam_calib['rotation'])
                    )
            
                    # Store Camera -> Ego (project_lidar_to_image expects Camera->World and inverts it)
                    # So we provide Camera->Ego, which when inverted gives Ego->Camera for projection
                    calibration_dict[cam_channel] = {
                        'intrinsic': np.array(cam_calib['camera_intrinsic']),
                        'extrinsic': np.linalg.inv(T_cam_to_ego) # Camera -> Ego (will be inverted by project_lidar_to_image)
                    }

        return FrameData(
            frame_index=idx,
            timestamp=sample['timestamp'] / 1e6,
            point_cloud=points, # Now in Ego Frame (N, 4)
            images=images_dict,
            metadata={
                'calibration': calibration_dict,
                'token': sample['token'],
                'gt_boxes': gt_boxes
            }
        )
    
    def _get_single_sweep_strict(self, key_token: str):
        """
        Directly loads the .pcd.bin file to guarantee NO ghosting/accumulation.
        """
        key_data = self.nusc.get('sample_data', key_token)
        calib = self.nusc.get('calibrated_sensor', key_data['calibrated_sensor_token'])
        
        # 1. Load File Directly
        pcl_path = Path(self.root) / key_data['filename']
        pc = LidarPointCloud.from_file(str(pcl_path))
        
        # 2. Get Transform (Sensor -> Ego)
        T_sensor_to_ego = transform_matrix(calib['translation'], Quaternion(calib['rotation']))

        # 3. Apply Transform
        xyz = pc.points[:3, :]
        if pc.points.shape[0] >= 4:
            i = pc.points[3, :]
        else:
            i = np.zeros(xyz.shape[1])

        xyz_h = np.vstack((xyz, np.ones((1, xyz.shape[1]))))
        xyz_ego = T_sensor_to_ego @ xyz_h
        
        points_ego = np.vstack((xyz_ego[:3, :], i)).T
        return points_ego, T_sensor_to_ego
    
    @property
    def calibration(self):
        # Calibration is dynamic per frame in NuScenes (Ego moves), 
        # so we rely on metadata in get()
        return {}