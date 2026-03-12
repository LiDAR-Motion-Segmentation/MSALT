import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
from src.data.structures import FrameData, SensorConfig
from src.core.objects import BoundingBox3D

class SemanticKittiLoader:
    """
    Loader for SemanticKITTI (Odometry Layout).
    - Reads .bin point clouds.
    - Reads .label binary files.
    - Generates GT Bounding Boxes from Instance IDs.
    """
    def __init__(self, config: SensorConfig):
        self.root = Path(config.lidar_path)
        
        # Handle sequence ID padding (0 -> "00")
        seq_str = str(config.extra_params.get("scenes", ["00"])[0]).zfill(2)
        
        self.seq_path = self.root / seq_str
        self.lidar_dir = self.seq_path / "velodyne"
        self.label_dir = self.seq_path / "labels"
        self.calib_path = self.seq_path / "calib.txt"
        
        if not self.lidar_dir.exists():
          raise FileNotFoundError(f"Velodyne folder not found at {self.lidar_dir}")
        
        self.files = sorted(list(self.lidar_dir.glob("*.bin")))
        
        self.available_cameras = {} # {'cam_2': Path(...), 'cam_3': Path(...)}
        for i in range(4):
            cam_dir = self.seq_path / f"image_{i}"
            if cam_dir.exists():
                self.available_cameras[f"cam_{i}"] = cam_dir
                
        self.calib = self._parse_calib(self.calib_path)
        
        # Map integer IDs to names (simplified for internal logic)
        # Class ID Mapping (SemanticKITTI -> Label)
        self.id_to_name = {
            10: "Car", 11: "Cyclist", 13: "Bus", 15: "Truck", 
            30: "Pedestrian", 31: "Cyclist", 32: "Cyclist",
            252: "Car", 253: "Cyclist", 254: "Pedestrian", 
            255: "Cyclist", 256: "Car", 257: "Bus", 258: "Truck"
        }
      
    def __len__(self):
        return len(self.files)
      
    def get_camera_ids(self):
        """ The UI calls this to create the view widgets."""
        return list(self.available_cameras.keys())
      
    def get(self, idx: int) -> FrameData:
        bin_path = self.files[idx]
        
        # KITTI points are x,y,z remission (float32)
        points = np.fromfile(bin_path, dtype = np.float32).reshape(-1, 4)
        
        images = {}
        for cam_id, cam_dir in self.available_cameras.items():
            img_path = cam_dir / f"{bin_path.stem}.png"
            if img_path.exists():
                # MSALT expects numpy array (H, W, 3)
                images[cam_id] = np.array(Image.open(img_path))
          
        # Generate GT Boxes from Semantic Labels  
        label_path = self.label_dir / f"{bin_path.stem}.label"
        gt_boxes = []
        if label_path.exists():
            gt_boxes = self._generate_boxes_from_sem_labels(points, label_path)
            
        return FrameData(
          frame_index=idx,
          point_cloud=points,
          images=images,
          metadata={
            "gt_boxes": gt_boxes,
            "calibration": self.calib,
            "file_id": bin_path.stem
          }
        )
    
    def _generate_boxes_from_sem_labels(self, points, label_path):
        """
        Parses binary label file, groups by instance ID, and fits boxes.
        """
        
        # Load Label Data (uint32)
        # Lower 16 bits = Semantic Class, Upper 16 bits = Instance ID
        label_data = np.fromfile(label_path, dtype=np.uint32)
        
        if len(label_data) != len(points):
            return [] # Mismatch safety check
          
        sem_labels = label_data & 0xFFFF
        inst_labels = label_data >> 16
        
        unique_instances = np.unique(inst_labels)
        boxes = []
        
        for inst_id in unique_instances:
            if inst_id == 0:
                continue # 0 is usually background/stuff
              
            # get points for this instance
            mask = inst_labels == inst_id
            inst_points = points[mask, :3]
            
            # Get Semantic Class for this instance (take the most frequent one)
            cls_id = sem_labels[mask][0]  
            
            # filter: We only care about dynamic objects (Cars, Peds, etc.)
            label_name = self.id_to_name.get(cls_id)
            if label_name is None:
                continue
              
            if len(inst_points) < 5:
                continue
              
            # Fit Oriented Bounding Box
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(inst_points)
            
            try:
              obb = pcd.get_oriented_bounding_box()
              center = obb.center
              extent = obb.extent
              R = obb.R
                
              # Extract Heading (Yaw) from Rotation Matrix
              # Simplified: assumes object is mostly upright
              heading = np.arctan2(R[1, 0], R[0, 0])
              
              boxes.append(BoundingBox3D(
                x=center[0], y=center[1], z=center[2],
                dx=extent[0], dy=extent[1], dz=extent[2],
                heading=heading,
                label=label_name,
                track_id=int(inst_id)
              ))
            except Exception:
              continue
            
        return boxes
      
    def _parse_calib(self, calib_path):
        """
        Parses KITTI calib.txt to get Tr_velo_to_cam and Rectification.
        """
        
        data = {}
        if not calib_path.exists(): 
          return {}
        
        with open(calib_path, 'r') as f:
            for line in f:
                if not line.strip(): 
                  continue
                key, val = line.split(':', 1)
                data[key] = np.array([float(x) for x in val.split()]).reshape(3, 4)
        
        # Base Transformation: Lidar -> Rectified Cam 0
        Tr = np.eye(4)
        if 'Tr' in data: 
            Tr[:3, :4] = data['Tr']
        elif 'Tr_velo_to_cam' in data: 
            Tr[:3, :4] = data['Tr_velo_to_cam']
        
        R0 = np.eye(4)
        if 'R0_rect' in data:
            R0[:3, :3] = data['R0_rect'][:3, :3].reshape(3,3)
            
        T_lidar_to_cam0_rect = R0 @ Tr
        
        final_calib = {}
        
        # Calculate for each detected camera
        for cam_id in self.available_cameras.keys():
            # Get P matrix (e.g., P2 for cam_2)
            p_key = f"P{cam_id.split('_')[-1]}"
            if p_key not in data: 
              continue
            
            P = data[p_key]
            
            # Decompose P = K * [I | t]
            # K is 3x3
            K = P[:3, :3]
            
            # Calculate Baseline Translation relative to Cam 0
            # P[0, 3] = fx * tx  => tx = P[0, 3] / fx
            fx = P[0, 0]
            tx = P[0, 3] / fx
            ty = P[1, 3] / P[1, 1] # Should be 0 usually
            
            # Transform Cam 0 -> Cam X
            # Since Cam X is at x=tx relative to Cam 0, the transform POINT_0 -> POINT_X is translation by (tx, ty, 0)
            T_cam0_to_camX = np.eye(4)
            T_cam0_to_camX[0, 3] = tx
            T_cam0_to_camX[1, 3] = ty

            # Full Extrinsic: Lidar -> Cam X
            T_lidar_to_camX = T_cam0_to_camX @ T_lidar_to_cam0_rect

            final_calib[cam_id] = {
                "intrinsic": K,
                "extrinsic": T_lidar_to_camX
            }
            
        return final_calib