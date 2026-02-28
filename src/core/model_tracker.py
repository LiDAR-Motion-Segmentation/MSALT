import numpy as np
import logging
import os
import tensorflow as tf
from src.core.objects import BoundingBox3D
from src.core.geometry import GeometryUtils

logger = logging.getLogger(__name__)

class DeepBoxTracker:
    def __init__(self, model_path: str) -> None:
        self.model = None
        self.history_buffer: list = []  # Stores [x, y, z, heading]
        self.SEQUENCE_LENGTH = 5  # need to take this from config + need to check from SUSTechPoints tool
            
        self._setup_tf_memory_growth()
        
        logger.info(f"Loading 3D Rotation Model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tracker model not found at {model_path}")
            
        self.rotation_model = tf.keras.models.load_model(model_path)
        
        # Warmup the model so the first click in the UI doesn't lag
        dummy_input = np.zeros((1, 512, 3), dtype=np.float32)
        _ = self.rotation_model(dummy_input, training=False)
        logger.info("3D Rotation Model loaded and warmed up.")
        
    def _setup_tf_memory_growth(self):
        """CRITICAL: Force TF to use CPU to prevent deadlocking PyQt's OpenGL renderer."""
        try:
            # Hide all GPUs from TensorFlow so it doesn't fight the UI
            tf.config.set_visible_devices([], 'GPU')
            logger.info("Forced TensorFlow to use CPU for UI stability.")
        except RuntimeError as e:
            logger.warning(f"TF Device Configuration Error: {e}")
                
    def _sample_points(self, points: np.ndarray, num_points: int = 512) -> np.ndarray:
        """Pads or downsamples points to exactly match model input shape."""
        num_current = points.shape[0]
        if num_current == 0:
            return np.zeros((num_points, 3), dtype=np.float32)
        
        if num_current < num_points:
            # pad with random existing points rather than zeros for better geometry
            idx = np.random.choice(num_current, num_points - num_current, replace = True)
            padding = points[idx]
            return np.vstack((points, padding))
        else:
            # Random downsample
            idx = np.random.choice(num_current, num_points, replace=False)
            return points[idx]
        
    def track(self, previous_box: BoundingBox3D, current_points: np.ndarray) -> BoundingBox3D:
        """
        Tracks a bounding box into the current frame.

        Args:
            previous_box (BoundingBox3D)
            current_points (np.ndarray)

        Returns:
            BoundingBox3D
        """
        if current_points is None or len(current_points) == 0:
            return previous_box
        
        # CROP: Create a "Search Region" (Expand the previous box by 1.5 meters)
        search_box = BoundingBox3D(
            x=previous_box.x, 
            y=previous_box.y, 
            z=previous_box.z,
            dx=previous_box.dx + 1.5, 
            dy=previous_box.dy + 1.5, 
            dz=max(previous_box.dz, 2.0), # Ensure enough height for Z-variance
            heading=previous_box.heading,
            label=previous_box.label,
            track_id=previous_box.track_id
        )
        
        point_indices = GeometryUtils.get_points_in_box(current_points, search_box)
        if len(point_indices) < 5:
            logger.warning(f"Tracker lost ID {previous_box.track_id}: Not enough points.")
            return previous_box # Return the old box if object is occluded/lost
        
        cropped_points = current_points[point_indices][:, :3]
        
        # predict yaw: center the points and pass to model
        centroid = np.mean(cropped_points, axis=0)
        centered_pts = cropped_points - centroid
        
        sampled_pts = self._sample_points(centered_pts, 512)
        model_input = tf.convert_to_tensor(sampled_pts.reshape(1, 512, 3), dtype=tf.float32)
        
        # use model(inputs) instead of model.predict() for real time speed
        pred_val = self.rotation_model(model_input, training=False)
        pred_cls = np.argmax(pred_val.numpy(), axis=-1)[0]
        
        # original math: (class * 3 + 1.5) degrees
        new_heading = (pred_cls * 3.0 + 1.5) * np.pi / 180.0
        
        # fit box: rotate points by negative yaw to align with axes, then AABB
        cos_y = np.cos(-new_heading)
        sin_y = np.sin(-new_heading)
        rot_mat = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [    0,      0, 1]
        ])
        
        # apply inverse rotation to make the object perfectly axis-aligned
        local_pts = np.dot(centered_pts, rot_mat.T)
        
        pmin = np.min(local_pts, axis=0)
        pmax = np.max(local_pts, axis=0)
        
        new_dims = pmax - pmin
        
        # Protect against flat dimensions
        new_dx = max(new_dims[0], 0.1)
        new_dy = max(new_dims[1], 0.1)
        new_dz = max(new_dims[2], 0.1)
        
        local_center = (pmax + pmin) / 2.0
        
        # rotate center back to global coordinates using positive yaw
        inv_rot_mat = np.array([
            [ cos_y, sin_y, 0],
            [-sin_y, cos_y, 0],
            [     0,     0, 1]
        ])
        global_center_offset = np.dot(local_center, inv_rot_mat.T)
        new_center = centroid + global_center_offset
        
        # Return perfectly wrapped BoundingBox
        return BoundingBox3D(
            x=new_center[0],
            y=new_center[1],
            z=new_center[2],
            dx=new_dx,
            dy=new_dy,
            dz=new_dz,
            heading=new_heading,
            label=previous_box.label,
            track_id=previous_box.track_id
        )
        
    def push_state(self, box: BoundingBox3D):
        """Adds a box state to the history buffer."""
        state = [box.x, box.y, box.z, box.heading]
        self.history_buffer.append(state)
        
        # keep buffer fixed size
        if len(self.history_buffer) > self.SEQUENCE_LENGTH:
            self.history_buffer.pop(0)
            
    def predict_next(self):
        """
        Returns [x, y, z, h] for the next frame.
        """
        
        # Input Shape: (1, Sequence_Length, 4)
        # We might need to normalize these inputs (e.g. subtract mean) depending on training
        input_seq = np.array([self.history_buffer], dtype=np.float32)
        
        try:
            # Inference
            # Outputs is usually offsets [dx, dy, dz, dh]
            prediction = self.model.predict(input_seq, verbose=0)
            offsets = prediction[0]
            
            last_state = self.history_buffer[-1]
            next_state = [
                last_state[0] + offsets[0], # x + dx
                last_state[1] + offsets[1], # y + dy
                last_state[2] + offsets[2], # z + dz
                last_state[3] + offsets[3]  # h + dh
            ]
            return next_state
        
        except Exception as e:
            logger.error(f"Inference error: {e}")