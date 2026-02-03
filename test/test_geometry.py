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


def test_interpolate_box_midpoint():
    """Interpolation at t=0.5 should give geometric midpoint and averaged heading."""
    box_start = BoundingBox3D(x=0, y=0, z=0, dx=2, dy=2, dz=2, heading=0.0)
    box_end = BoundingBox3D(x=2, y=2, z=2, dx=4, dy=4, dz=4, heading=np.pi / 2)

    params = GeometryUtils.interpolate_box(box_start, box_end, t=0.5)

    assert np.isclose(params["x"], 1.0)
    assert np.isclose(params["y"], 1.0)
    assert np.isclose(params["z"], 1.0)
    assert np.isclose(params["dx"], 3.0)
    assert np.isclose(params["dy"], 3.0)
    assert np.isclose(params["dz"], 3.0)
    # Heading halfway between 0 and pi/2 (no wrap-around issues here)
    assert np.isclose(params["heading"], np.pi / 4)


def test_refine_heading_returns_current_when_too_few_points():
    """If there are fewer than 5 points, refine_heading should be a no-op."""
    points = np.zeros((3, 3))  # fewer than 5
    current_heading = 1.23

    new_heading = GeometryUtils.refine_heading(points, current_heading)
    assert np.isclose(new_heading, current_heading)
