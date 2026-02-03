import numpy as np
from shapely.geometry import Polygon
from typing import Dict, Optional

class GeometryUtils:
    """
    Stateless utility class for geometric calculations.
    """
    
    @staticmethod
    def create_box_polygon(box: Dict) -> Optional[Polygon]:
        """
        Converts a dictionary-based box into a Shapely Polygon.
        
        Args:
            box: Dict with keys {'x', 'y', 'dx', 'dy', 'heading'}
                 x, y: Center position (meters)
                 dx, dy: Dimensions (meters). dx is length (along heading), dy is width.
                 heading: Yaw angle in radians (counter-clockwise from East/X-axis).
        
        Returns:
            shapely.geometry.Polygon: The 2D footprint of the box.
            None: If the box dimensions are invalid.
        """
        try:
            # rotation matrix (2D)
            c = np.cos(box['heading'])
            s = np.sin(box['heading'])
            R = np.array([[c, -s],[s, c]])
            
            # Define Local Corners (Centered at 0,0)
            # NuScenes/KITTI convention: dx usually length, dy usually width
            dx_2 = box['dx'] / 2.0
            dy_2 = box['dy'] / 2.0
            
            # Clockwise or or CCW order doesn't matter for Shapely,
            corners_local = np.array([
                [dx_2, dy_2],   # Front Left
                [dx_2, -dy_2],  # Front Right
                [-dx_2, -dy_2], # Back Right
                [-dx_2, dy_2]   # Back Left
            ])
            
            # Rotate and Translate to global coordinates
            # Equation: P_global = R * P_local + Center
            corners_global = (R @ corners_local.T).T + np.array([box['x'], box['y']])
            
            return Polygon(corners_global)
        
        except Exception:
            return None
    
    @staticmethod
    def calculate_bev_iou(box_a: dict, box_b: dict) -> float:
        """
        Calculates Bird's Eye View (BEV) Intersection over Union.
        
        Args:
            box_a_dict: dict with keys {'x', 'y', 'dx', 'dy', 'heading'}
            box_b_dict: dict with keys {'x', 'y', 'dx', 'dy', 'heading'}
            
        Returns:
            float: IoU value between 0.0 and 1.0.
        """
        
        poly_a = GeometryUtils.create_box_polygon(box_a)
        poly_b = GeometryUtils.create_box_polygon(box_b)
        
        # Safety Check: If either polygon failed to create (e.g. NaN values)
        if poly_a is None or poly_b is None:
            return 0.0

        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0
        
        # Optimization: Quick bounding box check before expensive polygon intersection
        # If their envelopes don't intersect, the polygons definitely don't.
        if not poly_a.envelope.intersects(poly_b.envelope):
            return 0.0
        
        try:
            inter_area = poly_a.intersection(poly_b).area
            union_area = poly_a.area + poly_b.area - inter_area
            
            if union_area <= 1e-6:
                return 0.0
            
            return inter_area / union_area
        
        except Exception:
            # Fallback for topological errors (rare in shapely but possible)
            return 0.0