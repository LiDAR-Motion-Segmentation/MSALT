from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from src.core.objects import BoundingBox3D
    from src.data.structures import FrameData
    from src.ui.components.camera_view import CameraStripWidget
    from src.ui.components.lidar_view import LidarVisualizer

    UI_DEPS_AVAILABLE = True
except Exception:
    UI_DEPS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not UI_DEPS_AVAILABLE,
    reason="UI tests require PyQt6 and pyqtgraph",
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class DummyScatter:
    def __init__(self):
        self.calls = []

    def setData(self, **kwargs):
        self.calls.append(kwargs)


class DummyViewWidget:
    def __init__(self):
        self.overlay_boxes = []
        self.removed = []
        self.added = []
        self.updated = False
        self.ground_z = 0.0

    def removeItem(self, item):
        self.removed.append(item)

    def addItem(self, item):
        self.added.append(item)

    def update(self):
        self.updated = True


def _make_box(label: str = "car") -> BoundingBox3D:
    return BoundingBox3D(
        x=0.0,
        y=0.0,
        z=0.0,
        dx=2.0,
        dy=2.0,
        dz=2.0,
        heading=0.0,
        label=label,
        track_id=1,
    )


def test_camera_strip_frame_update_sets_resolution_and_pixmap(qapp):
    widget = CameraStripWidget(["CAM_FRONT"])
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    frame = FrameData(frame_index=0, images={"CAM_FRONT": image})

    widget.on_frame_update(frame)

    label = widget.image_labels["CAM_FRONT"]
    assert label.orig_width == 12
    assert label.orig_height == 8
    assert label.pixmap() is not None


def test_camera_strip_box_signal_includes_shift_override(qapp, monkeypatch):
    widget = CameraStripWidget(["CAM_FRONT"])
    emissions = []
    widget.box_drawn.connect(lambda *args: emissions.append(args))

    # FIX: Replace QApplication in the module namespace rather than patching the C++ class directly
    class MockQApplication:
        @staticmethod
        def keyboardModifiers():
            return Qt.KeyboardModifier.ShiftModifier

    monkeypatch.setattr(
        "src.ui.components.camera_view.QApplication",
        MockQApplication,
    )
    
    widget._on_box_drawn("CAM_FRONT", 1, 2, 3, 4)

    assert emissions == [("CAM_FRONT", 1, 2, 3, 4, True)]


def test_camera_strip_update_3d_projection_updates_only_calibrated_camera(qapp):
    widget = CameraStripWidget(["CAM_FRONT", "CAM_BACK"])
    boxes = [_make_box()]
    intrinsic = np.eye(3)
    extrinsic = np.eye(4)

    widget.update_3d_projections(
        boxes,
        {"CAM_FRONT": {"intrinsic": intrinsic, "extrinsic": extrinsic}},
    )

    assert widget.image_labels["CAM_FRONT"].intrinsic is intrinsic
    assert widget.image_labels["CAM_FRONT"].extrinsic is extrinsic
    assert widget.image_labels["CAM_BACK"].intrinsic is None
    assert widget.image_labels["CAM_BACK"].extrinsic is None


def test_lidar_on_frame_update_sets_ground_plane_and_draws_when_no_boxes(qapp):
    visualizer = LidarVisualizer.__new__(LidarVisualizer)
    visualizer.current_points = None
    visualizer.ground_percentile = 50.0
    visualizer.ground_bias = 0.1
    visualizer.view_widget = DummyViewWidget()
    draw_calls = {"count": 0}

    def _draw_points_default():
        draw_calls["count"] += 1

    visualizer._draw_points_default = _draw_points_default

    points = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]])
    frame = FrameData(frame_index=0, point_cloud=points)
    visualizer.on_frame_update(frame)

    expected_ground = float(np.percentile(points[:, 2], 50.0) - 0.1)
    assert np.isclose(visualizer.view_widget.ground_z, expected_ground)
    assert draw_calls["count"] == 1


def test_lidar_draw_points_default_sets_scatter_data(qapp):
    visualizer = LidarVisualizer.__new__(LidarVisualizer)
    visualizer.scatter = DummyScatter()
    visualizer.current_points = np.array([[1.0, 2.0, -2.0], [3.0, 4.0, 3.0]])

    visualizer._draw_points_default()

    assert len(visualizer.scatter.calls) == 1
    call = visualizer.scatter.calls[0]
    assert np.array_equal(call["pos"], visualizer.current_points)
    assert call["size"] == 2
    assert call["color"].shape == (2, 4)


def test_lidar_update_boxes_recolors_points_and_rebuilds_line_items(qapp, monkeypatch):
    class DummyLineItem:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    visualizer = LidarVisualizer.__new__(LidarVisualizer)
    visualizer.current_points = np.array([[0.0, 0.0, 0.5], [10.0, 10.0, 0.5]])
    visualizer.scatter = DummyScatter()
    visualizer.view_widget = DummyViewWidget()
    visualizer.box_items = ["old-item"]
    visualizer.label_color_map = {"car": (1.0, 0.0, 0.0, 1.0)}

    monkeypatch.setattr(
        "src.ui.components.lidar_view.GeometryUtils.get_points_in_box",
        lambda points, box: np.array([0]),
    )
    monkeypatch.setattr("src.ui.components.lidar_view.gl.GLLinePlotItem", DummyLineItem)

    box = _make_box(label="car")
    visualizer.update_boxes([box])

    assert visualizer.view_widget.overlay_boxes == [box]
    assert visualizer.view_widget.removed == ["old-item"]
    assert len(visualizer.box_items) == 1
    assert len(visualizer.view_widget.added) == 1
    assert visualizer.view_widget.updated is True

    latest_scatter = visualizer.scatter.calls[-1]
    assert tuple(latest_scatter["color"][0]) == (1.0, 0.0, 0.0, 1.0)