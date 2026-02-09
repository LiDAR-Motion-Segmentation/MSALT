import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from src.core.objects import BoundingBox3D
from copy import deepcopy
from typing import Optional

class GeometryUtils:
    @staticmethod
    def project_lidar_to_image(
        xyz: np.ndarray, camera_pose: np.ndarray, intr: np.ndarray
    ):
        """
        Projects 3D LiDAR points onto the 2D Image Plane.
        
        Args:
            xyz: (N, 3) LiDAR points
            camera_pose: (4, 4) Matrix (Camera -> World/LiDAR)
            intr: (3, 3) Intrinsic Matrix
            
        Returns:
            uv: (N, 2) Projected pixel coordinates
            valid: (N,) Boolean mask (True if point is in front of camera)
        """
        if len(xyz) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=bool)
        
        # Add homogenous coordinates
        pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        
        # World -> Camera (Inverse of Pose)
        T_world_to_cam = np.linalg.inv(camera_pose)
        pts_cam = (T_world_to_cam @ pts_h.T).T[:, :3]
        
        # Filter Z > 0
        z = pts_cam[:, 2]
        
        # Small clip plane
        valid = z > 0 
        pts_cam_valid = pts_cam[valid]
        proj = (intr @ pts_cam_valid.T).T  # (M, 3)

        # Normalize: x/z, y/z
        uv_valid = proj[:, :2] / proj[:, 2:3]

        # Map back to full size array
        uv = np.zeros((xyz.shape[0], 2), dtype=np.float32)
        uv[valid] = uv_valid

        return uv, valid

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
        if len(points) == 0:
            return points

        # add 1 for homogenous coords
        N = points.shape[0]
        points_hom = np.hstack((points, np.ones((N, 1))))  # (N, 4)

        # Transform: (N, 4) @ (4, 4).T -> (N, 4)
        points_trans = points_hom @ T.T

        # return cartesian (x,y,z)
        return points_trans[:, :3]

    @staticmethod
    def get_frustum_points(
        points: np.ndarray, bbox_2d: tuple, K: np.ndarray, T_lidar_to_cam: np.ndarray
    ) -> np.ndarray:
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

        pts_valid = pts_cam[valid_z_mask]  # (M, 3)

        # Project to Image Plane (u, v)
        # [u, v, 1]^T = K * [x/z, y/z, 1]^T
        pts_pro = pts_valid @ K.T  # (N, 3)

        z_vals = pts_valid[:, 2]
        u = pts_pro[:, 0] / z_vals  # (M,)
        v = pts_pro[:, 1] / z_vals  # (M,)

        # Check BBox bounds
        box_x, box_y, box_w, box_h = bbox_2d

        in_u = (u >= box_x) & (u <= (box_x + box_w))
        in_v = (v >= box_y) & (v <= (box_y + box_h))

        # Result for the subset (M,)
        subset_mask = in_u & in_v

        # Map back to full size (N,)
        final_mask = np.zeros(len(points), dtype=bool)
        final_mask[valid_z_mask] = subset_mask

        return final_mask

    @staticmethod
    def _box_params_at_ray_depth(
        fallback_center: Optional[np.ndarray],
        std_dx: float,
        std_dy: float,
        std_dz: float,
        camera_heading: float,
    ) -> Optional[dict]:
        """
        Returns box params for a standard-sized box centered at ray-cast depth.
        Used when point cloud is too sparse or fitting fails (e.g. far objects).
        """
        if fallback_center is None:
            return None
        return {
            "x": float(fallback_center[0]),
            "y": float(fallback_center[1]),
            "z": float(fallback_center[2]) + (std_dz / 2),
            "dx": std_dx,
            "dy": std_dy,
            "dz": std_dz,
            "heading": camera_heading,
        }

    @staticmethod
    def fit_box_to_cloud(
        points: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 8,
        label: str = "unknown",
        camera_heading: float = 0.0,
        fallback_center: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Fits a box to the cloud.
        - If points < 5: uses ray-depth fallback when fallback_center is set.
        - If points >= 5: uses DBSCAN + OBB; on failure falls back to ray-depth when set.
        """
        DIMS = {
            "static_car": (4.5, 2.0, 1.6),
            "moving_car": (4.5, 2.0, 1.6),
            "moving_people": (0.5, 0.5, 1.7),
            "static_people": (0.5, 0.5, 1.7),
            "truck": (10.0, 2.5, 3.5),
            "bus": (12.0, 3.0, 3.5),
            "unknown": (1.0, 1.0, 1.0),
        }
        std_dx, std_dy, std_dz = DIMS.get(label, (1.0, 1.0, 1.0))

        if points is None or len(points) < 5:
            return GeometryUtils._box_params_at_ray_depth(
                fallback_center, std_dx, std_dy, std_dz, camera_heading
            )

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(points)

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if not unique_labels:
            return GeometryUtils._box_params_at_ray_depth(
                fallback_center, std_dx, std_dy, std_dz, camera_heading
            )

        # Heuristic: Pick the cluster closest to the sensor origin (0,0,0)
        best_cluster_pts = None
        min_dist = float("inf")

        for lbl in unique_labels:
            cluster_pts = points[labels == lbl]
            # Dist to origin
            dist = np.linalg.norm(np.mean(cluster_pts, axis=0))
            if dist < min_dist:
                min_dist = dist
                best_cluster_pts = cluster_pts

        if best_cluster_pts is None:
            return GeometryUtils._box_params_at_ray_depth(
                fallback_center, std_dx, std_dy, std_dz, camera_heading
            )

        # Use Open3D for robust OBB fitting
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(best_cluster_pts)

        try:
            obb = pcd.get_oriented_bounding_box()
        except RuntimeError:
            return GeometryUtils._box_params_at_ray_depth(
                fallback_center, std_dx, std_dy, std_dz, camera_heading
            )

        # Extract params
        center = obb.center
        extent = obb.extent
        R_mat = obb.R

        # Calculate yaw (heading) from Rotation Matrix
        # Heading is rotation around Z.
        # R = [[cos, -sin, 0], [sin, cos, 0], ..]
        heading = np.arctan2(R_mat[1, 0], R_mat[0, 0])

        # Apply your script's specific scaling offsets if you want exact parity
        dx = max(0.1, extent[0] - 0.59)  # Safety: prevent negative size
        dy = extent[1]
        dz = extent[2] + 1.17

        return {
            "x": center[0],
            "y": center[1],
            "z": center[2],
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "heading": heading,
        }

    @staticmethod
    def get_points_in_mask(
        points: np.ndarray, mask: np.ndarray, intr: np.ndarray, camera_pose: np.ndarray
    ) -> np.ndarray:
        """
        Selects 3D points that project onto the 'True' pixels of a 2D mask.
        Implements your 'mask_point_indices' logic.
        """
        # Project ALL points to image plane
        uv, valid = GeometryUtils.project_lidar_to_image(points, camera_pose, intr)
        h, w = mask.shape

        # Convert to integer pixels
        pixel_uv = np.round(uv).astype(int)

        # Check Image Bounds (Vectorized)
        # We only check points that were already valid (z > 0)
        in_img_bounds = (
            (pixel_uv[:, 0] >= 0)
            & (pixel_uv[:, 0] < w)
            & (pixel_uv[:, 1] >= 0)
            & (pixel_uv[:, 1] < h)
        )

        # Combine: Must be in front of cam AND inside image frame
        final_candidates_mask = valid & in_img_bounds

        # Check Mask Value
        # Extract u,v indices for candidate points
        valid_u = pixel_uv[final_candidates_mask, 0]
        valid_v = pixel_uv[final_candidates_mask, 1]

        # Check if mask is 1 at these coordinates
        # mask[v, u] because numpy is (row, col)
        is_in_mask = mask[valid_v, valid_u] == 1

        # Create Final 3D Mask
        # Start with all False
        mask_3d = np.zeros(len(points), dtype=bool)
        # Only set True for the candidates that passed the mask check
        mask_3d[final_candidates_mask] = is_in_mask

        return mask_3d

    @staticmethod
    def get_points_in_box(points: np.ndarray, box) -> np.ndarray:
        """
        Finds which LiDAR points are strictly inside a 3D Box.
        Used to 're-color' points (Red/Blue) after propagating.
        """
        if points is None or len(points) == 0:
            return np.array([], dtype=int)

        # Translate points to Box Frame
        center = np.array([box.x, box.y, box.z])
        pts_local = points - center

        # Rotate points to align with Box Axes
        rot_mat = R.from_euler("z", -box.heading).as_matrix()
        pts_aligned = pts_local @ rot_mat.T

        # Check Bounds (Is point inside the box dimensions?)
        half_dx, half_dy, half_dz = box.dx / 2.0, box.dy / 2.0, box.dz / 2.0

        mask = (
            (np.abs(pts_aligned[:, 0]) <= half_dx)
            & (np.abs(pts_aligned[:, 1]) <= half_dy)
            & (np.abs(pts_aligned[:, 2]) <= half_dz)
        )

        return np.where(mask)[0]

    @staticmethod
    def project_box_to_image(
        box, camera_pose: np.ndarray, intr: np.ndarray, image_shape
    ) -> Optional[list[int]]:
        """
        Projects a 3D Box to a 2D Bounding Rect [x, y, w, h] for SAM prompting.
        """
        # Get 8 Corners of the 3D Box
        corners_3d = box.get_corners()

        # Project 3D Corners -> 2D Pixels
        uv, valid = GeometryUtils.project_lidar_to_image(corners_3d, camera_pose, intr)

        # If fewer than 4 corners are visible, the object is likely off-screen
        if np.sum(valid) < 4:
            return None

        u_valid = uv[valid, 0]
        v_valid = uv[valid, 1]

        # Clip to Image Dimensions
        h, w = image_shape[:2]
        
        # Clamp to image
        min_x = np.clip(np.min(u_valid), 0, w)
        max_x = np.clip(np.max(u_valid), 0, w)
        min_y = np.clip(np.min(v_valid), 0, h)
        max_y = np.clip(np.max(v_valid), 0, h)

        width = max_x - min_x
        height = max_y - min_y

        if width <= 5 or height <= 5:
            return None

        return [int(min_x), int(min_y), int(width), int(height)]

    @staticmethod
    def interpolate_box(box_start, box_end, t: float):
        """
        Interpolates between two BoundingBox3D objects.
        t: 0.0 (start) to 1.0 (end)
        """
        
        # linear interpolation for position & dimensions
        x = box_start.x + (box_end.x - box_start.x) * t
        y = box_start.y + (box_end.y - box_start.y) * t
        z = box_start.z + (box_end.z - box_start.z) * t
        
        dx = box_start.dx + (box_end.dx - box_start.dx) * t
        dy = box_start.dy + (box_end.dy - box_start.dy) * t
        dz = box_start.dz + (box_end.dz - box_start.dz) * t
        
        rot_diff = box_end.heading - box_start.heading
        
        # Normalize diff to [-pi, pi]
        rot_diff = (rot_diff + np.pi) % (2 * np.pi) - np.pi
        heading = box_start.heading + rot_diff * t
        
        return {
            'x': x, 
            'y': y, 
            'z': z,
            'dx': dx, 
            'dy': dy, 
            'dz': dz,
            'heading': heading,
            'label': box_start.label,    
            'track_id': box_start.track_id
        }
        
    @staticmethod
    def refine_heading(points: np.ndarray, current_heading: float) -> float:
        """
        Uses PCA (Principal Component Analysis) to find the dominant axis 
        of the point cloud and align the heading.
        
        Args:
            points: (N, 3) points inside the box.
            current_heading: The current yaw estimate (to fix 180-degree ambiguity).
        
        Returns:
            new_heading: refined yaw in radians.
        """
        if len(points) < 5:
            return current_heading # Not enough points to be reliable
        
        # project to ground plane
        xy_points = points[:, :2]
        
        # Center the points first to remove translation
        mean = np.mean(xy_points, axis=0)
        centered = xy_points - mean
        
        # cov = (X.T @ X) / (N-1)
        cov = np.cov(centered.T)
        
        # eigenvalues = magnitude of variance
        # eigenvectors = direction of variance
        evals, evecs = np.linalg.eig(cov)
        
        # evecs are columns. evecs[:, i] corresponds to eval[i]
        idx = np.argmax(evals)
        major_axis = evecs[:, idx] # [dx, dy] vector
        
        # calculate angle from vector
        pca_heading = np.arctan2(major_axis[1], major_axis[0])
        
        # normalize diff to [-pi, pi] to check alignment
        diff = pca_heading - current_heading
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        if abs(diff) > (np.pi / 2):
            pca_heading += np.pi
            
        # normalize final results to standard [-pi, pi] range
        pca_heading = (pca_heading + np.pi) % (2 * np.pi) - np.pi
        
        return pca_heading
    
    @staticmethod
    def screen_to_ray(
        mouse_x: int,
        mouse_y: int,
        screen_w: int, 
        screen_h: int,
        view_matrix: np.ndarray,
        proj_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D screen coordinates to a 3D Ray (origin, direction)

        Args:
            mouse_x (int): Pixel coordinates (Top-Left origin)
            mouse_y (int): Pixel coordinates (Top-Left origin)
            screen_w (int): Viewport dimension
            screen_h (int): Viewport dimension
            view_matrix (np.ndarray): (4, 4) View Matrix (World -> Camera)
            proj_matix (np.ndarray): (4, 4) Projection Matrix (Camera -> Clip)

        Returns:
            tuple[np.ndarray, np.ndarray]: (3,) numpy array, (3,) normalized numpy array
        """
        # OpenGL NDC(Normalized Device Coordinates): x=[-1, 1], y=[-1, 1] (y is up)
        ndc_x = (2.0 * mouse_x) / screen_w - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y) / screen_h
        
        # clip near plan and far plane
        clip_near = np.array([ndc_x, ndc_y, -1.0, 1.0])
        clip_far = np.array([ndc_x, ndc_y, 1.0, 1.0])
        
        try:
            # We need to go from Clip Space -> World Space
            mvp = proj_matrix @ view_matrix
            inv_mvp = np.linalg.inv(mvp) 
        except np.linalg.LinAlgError:
            return np.zeros(3), np.array([0, 0, 1])
        
        # unproject
        res_near = inv_mvp @ clip_near
        res_far = inv_mvp @ clip_far
        
        # perspective divide
        if res_near[3] != 0:
            res_near /= res_near[3]
        if res_far[3] != 0:
            res_far /= res_far[3]
            
        # form ray
        origin = res_near[:3]
        end = res_far[:3]
        
        direction = end - origin
        norm = np.linalg.norm(direction)
        if norm > 1e-16:
            direction /= norm
            
        return origin, direction
    
    @staticmethod
    def intersect_ray_plane(ray_origin: np.ndarray, ray_dir: np.ndarray, plane_z: float = 0.0) -> np.ndarray:
        """
        Finds the intersection point of a ray and a horizontal plane at Z = plane_z.
         Start + t * Dir = Target
         RayZ + t * DirZ = PlaneZ
         t = (PlaneZ - RayZ) / DirZ
        """
        
        if abs(ray_dir[2]) < 1e-6:
            return None # ray is parallel to the plane
        
        t = (plane_z - ray_origin[2]) / ray_dir[2]
        
        if t < 0:
            return None # intersection is behind the camera
        
        return ray_origin + t * ray_dir
    
    @staticmethod
    def fit_box_with_pca(
        points: np.ndarray,
        previous_box: BoundingBox3D,
        eps: float = 0.5,
        min_samples: int = 4,
    ) -> BoundingBox3D:
        """
        Refines a box using DBSCAN (outlier removal) + PCA (orientation).
        """
        if len(points) < 5:
            return None
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        all_labels = clustering.labels_
        unique_labels = set(all_labels) - {-1}
        if not unique_labels:
            return None
        
        best_cluster = max(unique_labels, key=lambda cluster_id: np.sum(all_labels == cluster_id))
        inliers = points[all_labels == best_cluster]
        
        if len(inliers) < 5: 
            return None
        
        # PCA
        xy = inliers[:, :2]
        mean = np.mean(xy, axis=0)
        cov = np.cov((xy - mean).T)
        evals, evecs = np.linalg.eig(cov)
        major_axis = evecs[:, np.argmax(evals)]
        pca_heading = np.arctan2(major_axis[1], major_axis[0])
        
        # resolving ambiguity
        diff = pca_heading - previous_box.heading
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) > (np.pi / 2): 
            pca_heading += np.pi
        
        new_heading = (pca_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Fit Dimensions (Oriented)
        c, s = np.cos(-new_heading), np.sin(-new_heading)
        rot_mat = np.array([[c, -s], [s, c]])
        proj_xy = inliers[:, :2] @ rot_mat.T
        
        min_xy, max_xy = np.min(proj_xy, axis=0), np.max(proj_xy, axis=0)
        min_z, max_z = np.min(inliers[:, 2]), np.max(inliers[:, 2])

        new_dx = max_xy[0] - min_xy[0]
        new_dy = max_xy[1] - min_xy[1]
        new_dz = max_z - min_z

        center_pca = (min_xy + max_xy) / 2.0
        center_world = center_pca @ np.linalg.inv(rot_mat).T
        
        new_box = deepcopy(previous_box)
        new_box.x, new_box.y = center_world[0], center_world[1]
        new_box.z = min_z + (new_dz/2)
        new_box.dx, new_box.dy, new_box.dz = new_dx, new_dy, new_dz
        new_box.heading = new_heading
        
        return new_box
    
    @staticmethod
    def extrapolate_box(box: 'BoundingBox3D', velocity: dict, steps: int) -> 'BoundingBox3D':
        """
        Projects a box 'steps' frames into the future based on a velocity vector.
        velocity = {'x': float, 'y': float, 'z': float, 'heading': float}
        """
        
        # Linear projection
        new_x = box.x + (velocity['x'] * steps)
        new_y = box.y + (velocity['y'] * steps)
        new_z = box.z + (velocity['z'] * steps)
        
        # Heading projection (Simple addition, normalization happens usually at render time 
        # but good to normalize here to stay within -pi/pi)
        new_heading = box.heading + (velocity['heading'] * steps)
        while new_heading > np.pi: 
            new_heading -= 2 * np.pi
        while new_heading < -np.pi: 
            new_heading += 2 * np.pi

        return BoundingBox3D(
            track_id=box.track_id,
            label=box.label,
            x=new_x, 
            y=new_y, 
            z=new_z,
            dx=box.dx,  # Dimensions usually don't change
            dy=box.dy,
            dz=box.dz,
            heading=new_heading
        )