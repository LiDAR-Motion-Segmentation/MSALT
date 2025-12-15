from turtle import width
import pyqtgraph.opengl as gl
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout
from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData
from src.core.objects import BoundingBox3D


class LidarVisualizer(BasePluginWidget):
    def __init__(self):
        super().__init__(title="LiDAR 3D View")
        self._setup_ui()
        self.current_boxes = []

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view_widget = gl.GLViewWidget()
        self.view_widget.opts["distance"] = 20

        # grid
        grid = gl.GLGridItem()
        self.view_widget.addItem(grid)

        # Scatter Plot (The Point Cloud)
        self.scatter = gl.GLScatterPlotItem()
        self.view_widget.addItem(self.scatter)

        # Bounding Box Container (We will add/remove line items here)
        self.box_items = []

        layout.addWidget(self.view_widget)

    def on_frame_update(self, data: FrameData) -> None:
        if data.point_cloud is not None:
            points = data.point_cloud  # (N, 3)

            # Color Map Logic (Height Based)
            # Optimization: need to do this in C++ or use a pre-computed texture
            z = points[:, 2]
            colors = np.ones((points.shape[0], 4))
            colors[:, 0] = np.clip((z + 2) / 5, 0, 1)  # Red gradient
            colors[:, 1] = 0.5

            self.scatter.setData(pos=points, color=colors, size=2)

    def update_boxes(self, boxes: list[BoundingBox3D]):
        # clear old boxes
        for item in self.box_items:
            self.view_widget.removeItem(item)
        self.box_items.clear()

        # Draw new boxes
        # Connectivity for a cube wireframe (lines between corner indices)
        # Corners are 0-7.
        lines_indices = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # Bottom face
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # Top face
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # Vertical pillars
            ]
        )

        for box in boxes:
            corners = box.get_corners()  # (8, 3)

            # create line item
            line_item = gl.GLLinePlotItem(
                pos=corners, mode="lines", color=box.color, width=2, antialias=True
            )

            # GLLinePlotItem usually takes a list of points in sequence for 'lines' mode
            # Construct line pairs manually for GLLinePlotItem to be safe
            pts = []
            for start, end in lines_indices:
                pts.append(corners[start])
                pts.append(corners[end])
            pts = np.array(pts)

            line_item.setData(pos=pts, color=np.tile(box.color, (len(pts), 1)))

            self.view_widget.addItem(line_item)
            self.box_items.append(line_item)

    def reset(self):
        self.scatter.setData(pos=np.zeros((0, 3)))
