import numpy as np
import open3d as o3d

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
        if len(points) == 0: return points
        
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
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        
        # Transform LiDAR points to Camera Frame
        pts_cam = GeometryUtils.transform_points(points, T_lidar_to_cam)
        
        # Filter points BEHIND the camera (z <= 0)
        valid_z_mask = pts_cam[:, 2] > 0.1
        
        if not np.any(valid_z_mask):
            return np.zeros(len(points), dtype=bool)
        
        pts_valid = pts_cam[valid_z_mask] # (M, 3)
        
        # Project to Image Plane (u, v)
        # [u, v, 1]^T = K * [x/z, y/z, 1]^T
        pts_pro = pts_valid @ K.T # (N, 3)
    
        z_vals = pts_valid[:, 2]
        u = pts_pro[:, 0] / z_vals # (M,)
        v = pts_pro[:, 1] / z_vals # (M,)
        
        # Check BBox bounds
        box_x, box_y, box_w, box_h = bbox_2d
        
        in_u = (u >= box_x) & (u <= (box_x + box_w))
        in_v = (v >= box_y) & (v <= (box_y + box_h))
        
        # Result for the subset (M,)
        subset_mask = (in_u & in_v)
        
        # Map back to full size (N,)
        final_mask = np.zeros(len(points), dtype=bool)
        final_mask[valid_z_mask] = subset_mask
        
        return final_mask
    
    @staticmethod
    def fit_box_to_cloud(points: np.ndarray) -> dict:
        """
        Fits a box to the PRIMARY object in the point cloud.
        Uses DBSCAN clustering to separate the foreground object from background noise.
        """
        if len(points) < 5:
            return None
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Returns a list where labels[i] is the cluster ID of point i. -1 is noise.
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10, print_progress=False))
        
        if labels.max() == -1:
            print("[STATUS]: Clustering found only noise.")
        else:
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1] # Filter out noise
            
            best_cluster_points = points
            min_mean_dist = float('inf')
            
            for lbl in unique_labels:
                cluster_mask = (labels == lbl)
                cluster_pts = points[cluster_mask]

                # Calculate distance to origin (LiDAR/Camera position)
                # We use the centroid of the cluster.
                dist = np.linalg.norm(np.mean(cluster_pts, axis=0))
                
                if dist < min_mean_dist:
                    min_mean_dist = dist
                    best_cluster_points = cluster_pts
                    
            # Use the refined points
            points = best_cluster_points
            
        # Use percentiles to ignore outlier noise within the cluster itself
        min_p = np.percentile(points, 2, axis=0)
        max_p = np.percentile(points, 98, axis=0)
        
        center = (min_p + max_p) / 2
        dims = max_p - min_p
        
        # Sanity Check: Minimum size 10cm (prevents flat paper boxes)
        dims = np.maximum(dims, [0.1, 0.1, 0.1])
        
        # Create args for BoundingBox3D
        return {
            'x': center[0], 'y': center[1], 'z': center[2],
            'dx': dims[0], 'dy': dims[1], 'dz': dims[2],
            'heading': 0.0 # Default to 0 for now (AABB)
        }