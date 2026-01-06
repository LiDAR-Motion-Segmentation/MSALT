from PyQt6.QtGui import QKeyEvent
import pyqtgraph
import pyqtgraph.opengl as gl
import numpy as np
from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel, QVBoxLayout, 
                             QScrollArea, QFrame)
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
        self.points = points
        
        # Optimize View
        self.opts['distance'] = 6 # Zoom in close
        self.opts['center'] = pyqtgraph.Vector(box.x, box.y, box.z) # Center on object
        self.opts['azimuth'] = -90 # Look from behind/top usually
        self.opts['elevation'] = 30
        
        # Add Box Wireframe
        self._draw_scene()
        
    def _draw_scene(self):
        self.clear()
        
        # Points
        if len(self.points) > 0:
            colors = np.ones((len(self.points), 4))
            colors[:, :3] = 0.5 # Grey default
            
            # Highlight points inside the box
            indices = GeometryUtils.get_points_in_box(self.points, self.box)
            if len(indices) > 0:
                colors[indices] = np.array([1.0, 0.0, 0.0, 1.0]) # Red
                
            scatter = gl.GLScatterPlotItem(pos=self.points, size=3, color=colors)
            self.addItem(scatter)
        
        # box wireframe
        corners = self.box.get_corners()
        lines = np.array([
            [0,1], [1,2], [2,3], [3,0], # Bottom
            [4,5], [5,6], [6,7], [7,4], # Top
            [0,4], [1,5], [2,6], [3,7]  # Pillars
        ])
        pts = []
        for start, end in lines:
            pts.append(corners[start])
            pts.append(corners[end])
            
        # Color based on selection status
        line_item = gl.GLLinePlotItem(
            pos=np.array(pts), 
            mode='lines', 
            color=(0, 1, 0, 1), 
            width=2, 
            antialias=True
        )
        self.addItem(line_item)        
        
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
    
    def update_visuals(self):
        """Re-draws the scene (useful after moving the box)."""
        self._draw_scene()
        
    def mousePressEvent(self, ev):
        """Handle click to jump to frame."""
        self.clicked.emit(self.frame_idx)
        super().mousePressEvent(ev)
        
class BatchGridWindow(QWidget):
    request_jump = pyqtSignal(int)
    data_modified = pyqtSignal(int) 
    
    def __init__(self, data_controller, annotation_manager):
        super().__init__()
        self.data_ctrl = data_controller
        self.anno_mgr = annotation_manager
        
        self.setWindowTitle("Batch Correction (WASD to Move, QE to Rotate, RF to move up down)")
        self.resize(1200, 800)
        
        # Track Selection
        self.active_frame_idx = -1
        self.active_track_id = -1
        self.widgets_map = {} # Map frame_idx -> (ContainerWidget, MiniWidget)
        
        # Layouts
        self.main_layout = QVBoxLayout(self)
        
        # Header
        self.header = QLabel("Click a frame to edit. WASD=Move, QE=Rotate, RF=Up-Down")
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
        
    def load_track(self, track_id, start_frame_idx, window_size=16):
        """
        Loads crops for track_id from (center - window/2) to (center + window/2).
        """
        self.active_track_id = track_id
        self.active_frame_idx = start_frame_idx
        self.widgets_map.clear()
        
        # clears old widgets
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            
        if track_id == -1:
            self.header.setText("No Track Selected")
            return
        
        # Define Range
        start = start_frame_idx
        end = min(self.data_ctrl.get_total_frames(), start + window_size)
        
        self.header.setText(f"Track ID {track_id} | Checking Interpolation: Frames {start} - {end - 1}")
        
        # Columns for grid
        COLS = 6
        
        for i, f_idx in enumerate(range(start, end)):
            boxes = self.anno_mgr.get_boxes(f_idx)
            target = next((b for b in boxes if b.track_id == track_id), None)
            
            if not target:
                # If object lost, show empty placeholder or skip
                # lbl = QLabel(f"Frame {f_idx}: Lost")
                # lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # lbl.setStyleSheet("border: 1px solid #444; color: #888;")
                # row, col = divmod(i, COLS)
                # self.grid_layout.addWidget(lbl, row, col)
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
            
            container = QFrame()
            container.setFrameShape(QFrame.Shape.Box)
            container.setLineWidth(2)
            
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0,0,0,0)
            vbox.setSpacing(0)
            
            mini_widget = MiniFrameWidget(f_idx, crop_points, target)
            mini_widget.setMinimumSize(250, 250)
            # Connect Click
            mini_widget.clicked.connect(self.set_active_frame)
            
            lbl_info = QLabel(f"Frame {f_idx}")
            lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_info.setStyleSheet("background-color: #222; color: #eee;")
            
            vbox.addWidget(mini_widget)
            vbox.addWidget(lbl_info)
            
            row, col = divmod(i, COLS)
            self.grid_layout.addWidget(container, row, col)
            
            # Store references
            self.widgets_map[f_idx] = {
                'container': container,
                'widget': mini_widget,
                'label': lbl_info,
                'box': target
            }

        # Highlight the initial frame
        self.set_active_frame(start_frame_idx)
            
    def set_active_frame(self, frame_idx):
        """Highlights the selected frame and enables editing."""
        if frame_idx not in self.widgets_map:
            return
        
        self.active_frame_idx = frame_idx
        
        # update styles
        for f_idx, items in self.widgets_map.items():
            container = items['container']
            lbl = items['label']
            
            if f_idx == frame_idx:
                # SELECTED STYLE: Yellow Border & Text
                container.setStyleSheet("border: 3px solid #FFD700;") 
                lbl.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold;")
                
                # Scroll to make sure it's visible
                self.scroll.ensureWidgetVisible(container)
            else:
                # NORMAL STYLE
                container.setStyleSheet("border: 1px solid #444;")
                lbl.setStyleSheet("background-color: #222; color: #eee;")

        # Set Focus to this window so it catches Keys
        self.setFocus()
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle WASD (Move), QE (Rotate), RF (Up/Down)
        + SHIFT modifier for Scaling and TRIGGER AUTO-SAVE"""
        if self.active_frame_idx == -1 or self.active_frame_idx not in self.widgets_map:
            super().keyPressEvent(event)
            return
        
        items = self.widgets_map[self.active_frame_idx]
        box = items['box']
        widget = items['widget']
        
        STEP_MOVE = 0.1  # Meters
        STEP_ROT = 0.05  # Radians (~3 degrees)
        STEP_SCALE = 0.1  # Meters
        
        key = event.key()
        changed = False # Track if we actually changed anything
        modifiers = event.modifiers()
        is_shift = (modifiers & Qt.KeyboardModifier.ShiftModifier)
        
        # Translation (LiDAR Coords: X=Forward, Y=Left)
        # Z-AXIS (Height) CONTROL 
        # R / F for Up / Down
        if key == Qt.Key.Key_R:
            if is_shift:
                box.dz += STEP_SCALE # Taller
            else:
                box.z += STEP_MOVE   # Move Up
            changed = True
            
        elif key == Qt.Key.Key_F:
            if is_shift:
                box.dz = max(0.1, box.dz - STEP_SCALE) # Shorter (Prevent negative)
            else:
                box.z -= STEP_MOVE   # Move Down
            changed = True

        # X-AXIS (Length) CONTROL 
        # W / S for Forward / Backward
        elif key == Qt.Key.Key_W:
            if is_shift:
                box.dx += STEP_SCALE # Longer
            else:
                box.x += STEP_MOVE   # Move Forward
            changed = True
            
        elif key == Qt.Key.Key_S:
            if is_shift:
                box.dx = max(0.1, box.dx - STEP_SCALE) # Shorter
            else:
                box.x -= STEP_MOVE   # Move Backward
            changed = True

        # Y-AXIS (Width) CONTROL 
        # A / D for Left / Right
        elif key == Qt.Key.Key_A:
            if is_shift:
                box.dy += STEP_SCALE # Wider
            else:
                box.y += STEP_MOVE   # Move Left
            changed = True
            
        elif key == Qt.Key.Key_D:
            if is_shift:
                box.dy = max(0.1, box.dy - STEP_SCALE) # Thinner
            else:
                box.y -= STEP_MOVE   # Move Right
            changed = True

        # ROTATION (Yaw)
        # Q / E (Shift usually not needed for rotation speed, but possible)
        elif key == Qt.Key.Key_Q:
            box.heading += STEP_ROT
            changed = True
        elif key == Qt.Key.Key_E:
            box.heading -= STEP_ROT
            changed = True
            
        else:
            super().keyPressEvent(event)
            return
        
        if changed:
            widget.update_visuals()
            self.header.setText(f"Editing Frame {self.active_frame_idx} | Box: {box.x:.2f}, {box.y:.2f}")        
            
            # Emit the signal so MainWindow handles the file writing
            self.data_modified.emit(self.active_frame_idx)    
            
    def _handle_jump(self, f_idx):
        """Pass the signal up to Main Window."""
        self.request_jump.emit(f_idx)