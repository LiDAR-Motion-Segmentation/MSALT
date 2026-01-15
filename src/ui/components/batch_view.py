from PyQt6.QtGui import QKeyEvent
import pyqtgraph
import pyqtgraph.opengl as gl
import numpy as np
from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel, QVBoxLayout, 
                             QScrollArea, QFrame, QHBoxLayout, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
from src.core.geometry import GeometryUtils
from src.core.objects import BoundingBox3D
from copy import deepcopy
import math

class MiniFrameWidget(gl.GLViewWidget):
    """
    3D Viewer for a single frame. 
    Aligns the camera perfectly to the specific BoundingBox of this frame.
    """
    clicked = pyqtSignal(int) 
    
    def __init__(self, frame_idx, points, box: BoundingBox3D):
        super().__init__()
        self.frame_idx = frame_idx
        self.box = box
        self.points = points
        
        # State
        self.current_view_mode = "PERSPECTIVE"
        
        # Initial Draw
        self._draw_scene()
        
        # Set default view (Perspective)
        self.set_view_mode("PERSPECTIVE")
        
    def _draw_scene(self):
        self.clear()
        
        # Draw Points
        if len(self.points) > 0:
            colors = np.ones((len(self.points), 4))
            colors[:, :3] = 0.5 
            
            # Highlight points strictly inside the box
            indices = GeometryUtils.get_points_in_box(self.points, self.box)
            if len(indices) > 0:
                colors[indices] = np.array([1.0, 0.0, 0.0, 1.0]) # Red
                
            scatter = gl.GLScatterPlotItem(pos=self.points, size=2, color=colors)
            self.addItem(scatter)
        
        # Draw Box Wireframe (Green)
        corners = self.box.get_corners()
        
        # Define line connectivity (0-7 corners)
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

    def update_visuals(self):
        """Redraws scene and re-enforces alignment."""
        self._draw_scene()
        
        # Keep the view locked if we are in an aligned mode
        if self.current_view_mode in ["TOP", "SIDE", "FRONT"]:
            self.set_view_mode(self.current_view_mode)
        
    def mousePressEvent(self, ev):
        """Handle click to jump to frame."""
        self.clicked.emit(self.frame_idx)
        super().mousePressEvent(ev)
        
    def set_view_mode(self, mode: str):
        """
        Calculates the exact Camera Azimuth needed to make the Box appear 
        axis-aligned on screen, based on the specific heading of THIS frame's box.
        """
        self.current_view_mode = mode
        
        # Center precisely on the box
        self.opts['center'] = pyqtgraph.Vector(self.box.x, self.box.y, self.box.z)
        
        # Calculate Heading in Degrees
        # We negate it because PyQtGraph Azimuth rotates opposite to standard Math rotation
        h_deg = math.degrees(self.box.heading)
        
        # Calculate "Telephoto" Zoom Distance
        # To simulate Orthographic view with FOV=1, we need distance ~120x the object size.
        max_dim = max(self.box.dx, self.box.dy, self.box.dz)
        
        # Safety clamp for very small objects
        safe_dim = max(max_dim, 2.0)
        
        # We add extra padding (140x) so it's not "too zoomed in".
        telephoto_dist = safe_dim * 140.0 

        if mode == "TOP":
            # View from Top (Look down Z)
            # Alignment: We want Box Forward (+X) to point Up on Screen.
            # Azimuth -90 points Global Y Up. 
            # We subtract box heading to lock camera to box.
            self.opts['fov'] = 1 
            self.setCameraPosition(
                distance=telephoto_dist, 
                elevation=90, 
                azimuth= -90 - h_deg 
            )

        elif mode == "SIDE":
            # View from Side (Look along Y axis locally)
            # Alignment: We want to look at the "Right" side of the car.
            self.opts['fov'] = 1
            # -180 is often the "Right" side in PyQtGraph coords when Heading=0
            self.setCameraPosition(
                distance=telephoto_dist, 
                elevation=0, 
                azimuth= -180 - h_deg
            )

        elif mode == "FRONT":
            # View from Front (Look along X axis locally)
            # Alignment: Look at the "nose" or "tail".
            self.opts['fov'] = 1
            # -90 aligns with X axis
            self.setCameraPosition(
                distance=telephoto_dist, 
                elevation=0, 
                azimuth= -90 - h_deg
            )

        else: 
            # Perspective / Reset
            self.current_view_mode = "PERSPECTIVE"
            self.opts['fov'] = 60 
            self.setCameraPosition(distance=6, elevation=30, azimuth=-45 - h_deg)

class BatchGridWindow(QWidget):
    request_jump = pyqtSignal(int)
    data_modified = pyqtSignal(int) 
    
    def __init__(self, data_controller, annotation_manager):
        super().__init__()
        self.data_ctrl = data_controller
        self.anno_mgr = annotation_manager
        
        self.setWindowTitle("Batch Correction (WASD to Move, QE to Rotate, RF to move up down, Shift along with keys for scaling)")
        self.resize(1200, 800)
        
        # Track Selection
        self.active_frame_idx = -1
        self.active_track_id = -1
        self.widgets_map = {} # Map frame_idx -> (ContainerWidget, MiniWidget)
        
        self._setup_ui()
        
    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        
        # Header
        self.header = QLabel("Batch View: Controls align to the View!")
        self.header.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        self.main_layout.addWidget(self.header)
        
        # Toolbar
        self.toolbar_layout = QHBoxLayout()
        self.main_layout.addLayout(self.toolbar_layout)
        
        btn_top = self._create_btn("Top (XY)", "#1976D2")
        btn_side = self._create_btn("Side (XZ)", "#388E3C")
        btn_front = self._create_btn("Front (YZ)", "#E64A19")
        btn_reset = self._create_btn("3D View", "#455A64")
        
        self.toolbar_layout.addWidget(btn_top)
        self.toolbar_layout.addWidget(btn_side)
        self.toolbar_layout.addWidget(btn_front)
        self.toolbar_layout.addWidget(btn_reset)
        self.toolbar_layout.addStretch()
        
        btn_top.clicked.connect(lambda: self.change_all_views("TOP"))
        btn_side.clicked.connect(lambda: self.change_all_views("SIDE"))
        btn_front.clicked.connect(lambda: self.change_all_views("FRONT"))
        btn_reset.clicked.connect(lambda: self.change_all_views("RESET"))

        # Scroll / Grid
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(5)
        self.scroll.setWidget(self.grid_container)
        self.main_layout.addWidget(self.scroll)

    def _create_btn(self, text, color_hex):
        """Creates a styled button with manual hover color."""
        c = color_hex.lstrip('#')
        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        lighter = [min(255, int(x * 1.2)) for x in rgb]
        lighter_hex = f"#{lighter[0]:02x}{lighter[1]:02x}{lighter[2]:02x}"

        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color_hex}; color: white; font-weight: bold; 
                padding: 6px 12px; border-radius: 4px; border: 1px solid #333;
            }}
            QPushButton:hover {{ background-color: {lighter_hex}; }}
        """)
        return btn

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
            if item.widget(): item.widget().deleteLater()
            
        start = start_frame_idx
        end = min(self.data_ctrl.get_total_frames(), start + window_size)
        
        # Columns for grid
        COLS = 5
        
        for i, f_idx in enumerate(range(start, end)):
            # LOAD BOX SPECIFIC TO THIS FRAME
            boxes = self.anno_mgr.get_boxes(f_idx)
            target = next((b for b in boxes if b.track_id == track_id), None)
            
            if not target: continue
            
            frame_data = self.data_ctrl.get(f_idx)
            if frame_data.point_cloud is None: continue
            
            # Crop Points
            points = frame_data.point_cloud
            crop_box = deepcopy(target)
            crop_box.dx += 5.0; crop_box.dy += 5.0; crop_box.dz += 3.0
            indices = GeometryUtils.get_points_in_box(points, crop_box)
            crop_points = points[indices]
            
            # CREATE WIDGET WITH FRAME-SPECIFIC BOX 
            container = QFrame()
            container.setFrameShape(QFrame.Shape.Box)
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(0,0,0,0)
            
            mini_widget = MiniFrameWidget(f_idx, crop_points, target)
            mini_widget.setMinimumSize(220, 220)
            mini_widget.clicked.connect(self.set_active_frame)
            
            lbl = QLabel(f"Frame {f_idx}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            vbox.addWidget(mini_widget)
            vbox.addWidget(lbl)
            
            row, col = divmod(i, COLS)
            self.grid_layout.addWidget(container, row, col)
            self.widgets_map[f_idx] = {'container': container, 'widget': mini_widget, 'label': lbl, 'box': target}
            
        self.set_active_frame(start_frame_idx)

    def set_active_frame(self, frame_idx):
        if frame_idx not in self.widgets_map: return
        self.active_frame_idx = frame_idx
        
        for f_idx, items in self.widgets_map.items():
            if f_idx == frame_idx:
                items['container'].setStyleSheet("border: 3px solid #FFD700;") 
                items['label'].setStyleSheet("background-color: #FFD700; color: black;")
                self.scroll.ensureWidgetVisible(items['container'])
            else:
                items['container'].setStyleSheet("border: 1px solid #444;")
                items['label'].setStyleSheet("background-color: #222; color: #eee;")
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
            self.header.setText(f"Editing Frame {self.active_frame_idx} | Box: x={box.x:.3f}, y={box.y:.3f} z={box.z:.3f} | Scale: dx={box.dx:.3f}, dy={box.dy:.3f} dz={box.dz:.3f} | Heading: {box.heading}")
            
            # Emit the signal so MainWindow handles the file writing
            self.data_modified.emit(self.active_frame_idx)    
            
    def _handle_jump(self, f_idx):
        """Pass the signal up to Main Window."""
        self.request_jump.emit(f_idx)
        
    def change_all_views(self, mode: str):
        """Iterates over all active mini-widgets and changes their camera angle."""
        for items in self.widgets_map.values():
            widget = items['widget']
            widget.set_view_mode(mode)
            
        # Update Help Text
        if mode == "SIDE":
            self.header.setText("Side View (XZ). Controls: WASD move X/Z. (Y-axis hidden)")
        elif mode == "TOP":
            self.header.setText("Top View (XY). Controls: WASD move X/Y. (Z-axis hidden)")
        elif mode == "FRONT":
            self.header.setText("Front View (YZ). Controls: WASD move Z/Y. (X-axis hidden)")
        else:
            self.header.setText("3D View. WASD=Move, QE=Rotate")