from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict, List


@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    dx: float  # Length (x-axis)
    dy: float  # Width (y-axis)
    dz: float  # Height (z-axis)
    heading: float  # Yaw angle in radians

    # Metadata
    label: str = "Unknown"
    track_id: int = -1
    confidence: float = 1.0

    # State
    selected: bool = False
    point_indices: Optional[np.ndarray] = None

    source_2d: Optional[Dict] = None
    visual_overrides: Dict[str, List[int]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": self.confidence,
            "x": self.x, "y": self.y, "z": self.z,
            "dx": self.dx, "dy": self.dy, "dz": self.dz,
            "heading": self.heading,
        }
        # Only save if it exists to keep JSON clean
        if self.visual_override_2d:
            data["visual_override_2d"] = self.visual_override_2d
            
        if self.source_2d:
            data["source_2d"] = self.source_2d
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict):
        box = cls(
            x=data["x"], y=data["y"], z=data["z"],
            dx=data["dx"], dy=data["dy"], dz=data["dz"],
            heading=data["heading"],
            label=data.get("label", "Unknown"),
            track_id=data.get("track_id", -1),
            confidence=data.get("confidence", 1.0)
        )
        # Load overrides if they exist
        if "visual_override_2d" in data:
            box.visual_override_2d = data["visual_override_2d"]
            
        if "source_2d" in data:
            box.source_2d = data["source_2d"]
            
        return box

    def get_corners(self) -> np.ndarray:
        """
        Calculates the 8 corners of the box in world coordinates.
        Returns: (8, 3) numpy array
        """

        # dx=length, dy=width, dz=height
        x_corners = self.dx / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = self.dy / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = self.dz / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

        # Shape: (3, 8)
        corners = np.vstack((x_corners, y_corners, z_corners))

        # we rotate around Z axis (Yaw)
        rot_mat = R.from_euler("z", self.heading).as_matrix()  # (3, 3)
        corners = rot_mat @ corners

        # 3. Translate
        corners[0, :] += self.x
        corners[1, :] += self.y
        corners[2, :] += self.z

        return corners.T  # Return (8, 3)

    @property
    def color(self) -> Tuple[float, float, float, float]:
        if self.selected:
            return (1.0, 1.0, 0.0, 1.0)  # Yellow if selected
        return (0.0, 1.0, 0.0, 1.0)  # Green by default

    def __eq__(self, other) -> bool:
        """
        Custom equality check to handle NumPy arrays safely.
        """
        if not isinstance(other, BoundingBox3D):
            return False
        
        scalars_match = (
            self.track_id == other.track_id and
            self.label == other.label and
            np.isclose(self.x, other.x) and
            np.isclose(self.y, other.y) and
            np.isclose(self.z, other.z) and
            np.isclose(self.dx, other.dx) and
            np.isclose(self.dy, other.dy) and
            np.isclose(self.dz, other.dz) and
            np.isclose(self.heading, other.heading)
        )
        
        if not scalars_match:
            return False
        
        if self.point_indices is None and other.point_indices is None:
            return True
        if self.point_indices is None or other.point_indices is None:
            return False
            
        return np.array_equal(self.point_indices, other.point_indices)