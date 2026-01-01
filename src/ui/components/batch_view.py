import pyqtgraph
import pyqtgraph.opengl as gl
import numpy as np
from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel, QVBoxLayout, 
                             QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal

from src.core.geometry import GeometryUtils
from src.core.objects import BoundingBox3D
from copy import deepcopy

class MiniFrameWidget(gl.GLViewWidget):
    """
    A lightweight 3D viewer for a single frame crop.
    """
    clicked = pyqtSignal(int) 
    
    def __init__(self, frame_idx, points, box: BoundingBox3D):
        super().__init__()
        self.frame_idx = frame_idx
        self.box = box
        
        # Optimize View
        self.opts['distance'] = 6 # Zoom in close
        self.opts['center'] = pyqtgraph.Vector(box.x, box.y, box.z) # Center on object
        self.opts['azimuth'] = -90 # Look from behind/top usually
        self.opts['elevation'] = 30
        
        # adding points the crop
        self.scatter = gl.GLScatterPlotItem(pos=points, size=3)
        # color logic: Points inside box = Red, Outside = Grey
        if len(points) > 0:
            colors = np.ones((len(points), 4))
            # grey colour
            colors[:, 0] = 0.5 
            colors[:, 1] = 0.5
            colors[:, 2] = 0.5 
            
            # Red inside
            indices = GeometryUtils.get_points_in_box(points, box)
            if len(indices) > 0:
                colors[indices] = np.array([1.0, 0.0, 0.0, 1.0])
                
            self.scatter.setData(color=colors)
        self.addItem(self.scatter)
        
        # Add Box Wireframe
        self._add_box_visual()
        
    def _add_box_visual(self):
        corners = self.box.get_corners()
        # Wireframe indices
        lines = np.array([
            [0,1], [1,2], [2,3], [3,0], # Bottom
            [4,5], [5,6], [6,7], [7,4], # Top
            [0,4], [1,5], [2,6], [3,7]  # Pillars
        ])
        pts = []
        for start, end in lines:
            pts.append(corners[start])
            pts.append(corners[end])
            
        line_item = gl.GLLinePlotItem(
            pos=np.array(pts), 
            mode='lines', 
            color=(0, 1, 0, 1), # Green Box
            width=2, 
            antialias=True
        )
        self.addItem(line_item)
        
    def mousePressEvent(self, ev):
        """Handle click to jump to frame."""
        self.clicked.emit(self.frame_idx)
        super().mousePressEvent(ev)
        
class BatchGridWindow(QWidget):
    request_jump = pyqtSignal(int) 
    
    def __init__(self, data_controller, annotation_manager):
        super().__init__()
        self.data_ctrl = data_controller
        self.anno_mgr = annotation_manager
        
        self.setWindowTitle("Batch Track Gallery")
        self.resize(1200, 800)
        
        # Layouts
        self.main_layout = QVBoxLayout(self)
        
        # Header
        self.header = QLabel("Select a track in Main Window to view grid.")
        self.header.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        self.main_layout.addWidget(self.header)
        
        # Scroll Area for Grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(5)
        self.scroll.setWidget(self.grid_container)
        self.main_layout.addWidget(self.scroll)
        
    def load_track(self, track_id, center_frame, window_size=12):
        """
        Loads crops for track_id from (center - window/2) to (center + window/2).
        """
        # clears old widgets
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item.widget():
                w = item.widget()
                self.grid_layout.removeWidget(w)
                w.setParent(None)
                w.deleteLater()
            
        if track_id == -1:
            self.header.setText("No Track Selected")
            return
        
        # Define Range
        start = max(0, center_frame - (window_size // 2))
        end = min(self.data_ctrl.get_total_frames(), center_frame + (window_size // 2))
        
        self.header.setText(f"Track ID {track_id} | Frames {start} - {end}")
        
        # Columns for grid (e.g., 4 columns)
        # might want 5 in future lets see
        COLS = 4
        
        for i, f_idx in enumerate(range(start, end)):
            boxes = self.anno_mgr.get_boxes(f_idx)
            target = next((b for b in boxes if b.track_id == track_id), None)
            
            if not target:
                # If object lost, show empty placeholder or skip
                lbl = QLabel(f"Frame {f_idx}: Lost")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("border: 1px solid #444; color: #888;")
                row, col = divmod(i, COLS)
                self.grid_layout.addWidget(lbl, row, col)
                continue
            
            frame_data = self.data_ctrl.get(f_idx)
            if frame_data.point_cloud is None: 
                continue
            
            points = frame_data.point_cloud
            crop_box = deepcopy(target)
            crop_box.dx += 4.0 # 2m padding per side
            crop_box.dy += 4.0
            crop_box.dz += 2.0
            
            indices = GeometryUtils.get_points_in_box(points, crop_box)
            crop_points = points[indices]
            
            # create mini widgets
            widget = MiniFrameWidget(f_idx, crop_points, target)
            widget.clicked.connect(self._handle_jump)
            widget.setMinimumSize(250, 250)
            
            # Add Label Overlay
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0,0,0,0)
            vbox.setSpacing(0)
            
            lbl_info = QLabel(f"Frame {f_idx}")
            lbl_info.setStyleSheet("background-color: #222; color: #eee; padding: 2px;")
            lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            vbox.addWidget(widget)
            vbox.addWidget(lbl_info)
            
            # Highlight Current Frame
            if f_idx == center_frame:
                lbl_info.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold;")

            row, col = divmod(i, COLS)
            self.grid_layout.addWidget(container, row, col)
            
    def _handle_jump(self, f_idx):
        """Pass the signal up to Main Window."""
        self.request_jump.emit(f_idx)