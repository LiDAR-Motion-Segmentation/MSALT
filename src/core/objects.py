from dataclasses import dataclass, field
from turtle import heading
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple


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
