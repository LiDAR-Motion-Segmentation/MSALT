import numpy as np

class GeometryUtils:
    @staticmethod
    def pixel_to_ray(u: float, v: float, K_inv: np.ndarray) -> np.ndarray:
        """
        Converts a 2D pixel (u,v) into a 3D normalized ray vector in Camera Frame.
        Args:
            u, v: Pixel coordinates
            K_inv: Inverse of Intrinsic Matrix (3x3)
        Returns:
            (3,) Ray vector (x, y, z) where z=1 usually
        """
        
        # Homogenous pixel [u, v, 1]
        pixel = np.array([u, v, 1.0])
        
        # apply inverse intrinsics
        ray_cam = K_inv @ pixel
        
        return ray_cam
    
    @staticmethod
    def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Applies 4x4 Transformation Matrix to 3D points.
        Args:
            points: (N, 3)
            T: (4, 4) Transformation Matrix
        """
        
        # add 1 for homogenous coords
        N = points.shape[0]
        points_hom = np.hstack((points, np.ones((N, 1)))) # (N, 4)
        
        # Transform: (N, 4) @ (4, 4).T -> (N, 4)
        points_trans = points_hom @ T.T
        
        # return cartesian (x,y,z)
        return points_trans[:, :3]
    
    @staticmethod
    def get_frustum_points(points: np.ndarray,
                           bbox_2d: tuple,
                           K: np.ndarray,
                           T_lidar_to_cam: np.ndarray) -> np.ndarray:
        """
        Filters LiDAR points that project INSIDE a 2D bounding box.
        
        Args:
            points: (N, 3) Cloud in LiDAR Frame
            bbox_2d: (min_x, min_y, max_x, max_y)
            K: (3, 3) Intrinsic Matrix
            T_lidar_to_cam: (4, 4) Extrinsics
            
        Returns:
            mask: (N,) boolean array where True = inside box
        """
        # Transform LiDAR points to Camera Frame
        pts_cam = GeometryUtils.transform_points(points, T_lidar_to_cam)
        
        # Filter points BEHIND the camera (z <= 0)
        valid_z_mask = pts_cam[:, 2] > 0
        
        # Project to Image Plane (u, v)
        # [u, v, 1]^T = K * [x/z, y/z, 1]^T
        pts_pro = pts_cam @ K.T # (N, 3)
        
        # Normalize by Z
        # Avoid divide by zero using the valid mask
        u = np.zeros(points.shape[0])
        v = np.zeros(points.shape[0])
        
        z_valid = pts_pro[valid_z_mask, 2]
        u[valid_z_mask] = pts_pro[valid_z_mask, 0] / z_valid
        v[valid_z_mask] = pts_pro[valid_z_mask, 1] / z_valid
        
        # 4. Check BBox bounds
        min_x, min_y, max_x, max_y = bbox_2d
        
        in_u = (u >= min_x) & (u <= max_x)
        in_v = (v >= min_y) & (v <= max_y)
        
        return valid_z_mask & in_u & in_v
    
    @staticmethod
    def fit_box_to_cloud(points: np.ndarray) -> dict:
        """
        Fits a simple 3D box to a cluster of points.
        Returns kwargs for BoundingBox3D.
        """
        if len(points) == 0:
            return None
        
        # Use percentiles to ignore outlier noise (flying points)
        min_p = np.percentile(points, 5, axis=0) # 5th percentile
        max_p = np.percentile(points, 95, axis=0) # 95th percentile
        
        center = (min_p + max_p) / 2
        dims = max_p - min_p
        
        # Create args for BoundingBox3D
        return {
            'x': center[0], 'y': center[1], 'z': center[2],
            'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
            'heading': 0.0 # Default to 0 for now (AABB)
        }