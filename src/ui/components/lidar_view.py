import pyqtgraph.opengl as gl
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout
from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData

class LidarVisualizer(BasePluginWidget):
    def __init__(self):
        super().__init__(title="LiDAR 3D View")
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.view_widget = gl.GLViewWidget()
        self.view_widget.opts['distance'] = 20
        
        # grid
        grid = gl.GLGridItem()
        self.view_widget.addItem(grid)
        
        # Scatter Plot (The Point Cloud)
        self.scatter = gl.GLScatterPlotItem()
        self.view_widget.addItem(self.scatter)
        layout.addWidget(self.view_widget)
        
    def on_frame_update(self, data: FrameData) -> None:
        if data.point_cloud is None:
            return
        
        points = data.point_cloud # (N, 3)
        
        # Color Map Logic (Height Based)
        # Optimization: need to do this in C++ or use a pre-computed texture
        z = points[:, 2]
        colors = np.ones((points.shape[0], 4))
        colors[:, 0] = np.clip((z + 2) / 5, 0, 1) # Red gradient
        colors[:, 1] = 0.5
        
        self.scatter.setData(pos=points, color=colors, size=2)

    def reset(self):
        self.scatter.setData(pos=np.zeros((0,3)))