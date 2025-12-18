# tests/test_geometry.py
import numpy as np
from src.core.objects import BoundingBox3D
from src.core.geometry import GeometryUtils


def test_box_corners():
    """Test if a box generates 8 corners."""
    box = BoundingBox3D(x=0, y=0, z=0, dx=2, dy=2, dz=2, heading=0.0)
    corners = box.get_corners()
    assert corners.shape == (8, 3)


def test_points_in_box():
    """Test if points inside a box are correctly detected."""
    box = BoundingBox3D(x=0, y=0, z=0, dx=2, dy=2, dz=2, heading=0)

    # Create 3 points: Inside, Outside, Boundary
    points = np.array(
        [
            [0.5, 0.5, 0.5],  # Inside
            [5.0, 5.0, 5.0],  # Outside
            [0.0, 0.0, 0.0],  # Center (Inside)
        ]
    )

    indices = GeometryUtils.get_points_in_box(points, box)

    assert 0 in indices  # Point 0 should be inside
    assert 2 in indices  # Point 2 should be inside
    assert 1 not in indices  # Point 1 should be outside
