from PyQt6.QtGui import QPainter, QColor, QFont, QVector3D
from pyqtgraph.Qt.QtGui import QMatrix4x4
from PyQt6.QtCore import QRect, Qt, pyqtSignal 
import pyqtgraph.opengl as gl
import numpy as np
import logging
from PyQt6.QtWidgets import QVBoxLayout
from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData
from src.core.objects import BoundingBox3D
from src.core.geometry import GeometryUtils
from typing import List, Dict
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout

logger = logging.getLogger(__name__)

class DrawState:
    IDLE = 0
    DRAGGING_BASE = 1  # User is defining X/Y dimensions
    SETTING_HEIGHT = 2 # User is defining Z height

def get_projection_matrix(w: int, h: int, fov: float, distance: float) -> QMatrix4x4:
    """
    Calculates the projection matrix matching PyQtGraph's internal state.
    """
    matrix = QMatrix4x4()
    aspect = w / h if h > 0 else 1.0
    
    # Dynamic clipping planes based on camera distance to prevent z-fighting
    near_clip = max(distance * 0.001, 0.01)
    far_clip = distance * 1000.0
    
    matrix.perspective(fov, aspect, near_clip, far_clip)
    return matrix

class CustomGLWidget(gl.GLViewWidget):
    """
    Enhanced 3D Viewer with Text Overlay and Mouse Interaction.
    """
    # Signal: cx, cy, cz, dx, dy, dz, heading
    box_created = pyqtSignal(float, float, float, float, float, float, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data Storage
        self.overlay_boxes = []

        # Drawing State
        self.state = DrawState.IDLE
        self.draw_start_pt = None   # [x, y, z]
        self.draw_end_pt = None     # [x, y, z]
        self.draw_height = 1.5      # Default height
        self.ground_z = -1.5        # Assumed ground plane
        
        # Ghost Box (Visual Feedback while drawing)
        self.ghost_box = gl.GLBoxItem(color=(0, 255, 255, 255)) # Cyan
        self.ghost_box.setVisible(False)
        self.addItem(self.ghost_box)
        
        # This helps confirm if your Lidar is rotated correctly.
        axis = gl.GLAxisItem()
        axis.setSize(3, 3, 3)
        self.addItem(axis)
        
    def _get_matrices_np(self):
        """Helper to extract OpenGL matrices as Numpy arrays."""
        # View Matrix (World -> Camera)
        v_data = self.viewMatrix().data() # Tuple of 16 floats
        view_mat = np.array(v_data, dtype=np.float32).reshape(4, 4)
        
        # Projection Matrix (Camera -> Clip)
        w = self.width()
        h = self.height()
        fov = self.opts.get('fov', 60)
        dist = self.opts.get('distance', 20)
        
        qt_proj_mat = get_projection_matrix(w, h, fov, dist)
        p_data = qt_proj_mat.data()
        proj_mat = np.array(p_data, dtype=np.float32).reshape(4, 4)
        
        return view_mat.T, proj_mat.T
        
    def mousePressEvent(self, event):
        # using ctrl+left click start the process
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                logger.info(f"Ctrl+Click detected at {event.pos()}")
                if self.state == DrawState.IDLE:
                    vm, pm = self._get_matrices_np()
                    origin, direction = GeometryUtils.screen_to_ray(
                        event.pos().x(), 
                        event.pos().y(),
                        self.width(), 
                        self.height(),
                        vm, 
                        pm 
                    )
                    
                    hit = GeometryUtils.intersect_ray_plane(origin, direction, self.ground_z)
                    
                    if hit is not None:
                        logger.info("State changed to DRAGGING_BASE")
                        self.state = DrawState.DRAGGING_BASE
                        self.draw_start_pt = hit
                        self.draw_end_pt = hit
                        self.draw_height = 0.0 # start flat
                        self._update_ghost_box()
                    else:
                        logger.info("Ray missed the ground plane!")
                        
                elif self.state == DrawState.SETTING_HEIGHT:
                    # finish drawing
                    self._finalize_drawing()
                    self.state = DrawState.IDLE
                    self.ghost_box.setVisible(False)
            
            event.accept()
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        if self.state == DrawState.DRAGGING_BASE:
            vm, pm = self._get_matrices_np()
            origin, direction = GeometryUtils.screen_to_ray(
                event.pos().x(), 
                event.pos().y(),
                self.width(), 
                self.height(),
                vm, pm
            )
            
            hit = GeometryUtils.intersect_ray_plane(origin, direction, self.ground_z)
            
            if hit is not None:
                self.draw_end_pt = hit
                self._update_ghost_box()
                
        elif self.state == DrawState.SETTING_HEIGHT:
            dy = self.height() - event.pos().y() # Invert Y
            # Map screen Y to a reasonable height (0 to 5m)
            self.draw_height = max(0.5, (dy / 100.0))
            self._update_ghost_box()
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.state == DrawState.DRAGGING_BASE:
            # transition into height mode
            self.state = DrawState.SETTING_HEIGHT
            
            # setting a default height so that the box pops up immmediately
            self.draw_height = 1.6
            self._update_ghost_box()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def _update_ghost_box(self):
        """Updates the GLBoxItem to match current drawing state."""
        if self.draw_start_pt is None and self.draw_end_pt is None:
            return
        
        p1 = self.draw_start_pt
        p2 = self.draw_end_pt
        
        # calculate dimensions
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        dz = self.draw_height
        
        # caculate center, currently box item draws them from the center but translation needs to be done
        min_x = min(p1[0], p2[0])
        min_y = min(p1[1], p2[1])
        min_z = self.ground_z
        
        self.ghost_box.setSize(dx, dy, dz)
        self.ghost_box.resetTransform()
        self.ghost_box.translate(min_x, min_y, min_z)
        self.ghost_box.setVisible(True)
        self.update()
    
    def _finalize_drawing(self):
        p1 = self.draw_start_pt
        p2 = self.draw_end_pt
        
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        dz = self.draw_height
        
        cx = (p1[0] + p2[0]) / 2.0
        cy = (p1[1] + p2[1]) / 2.0
        cz = self.ground_z + (dz / 2.0)
        
        # min size check
        if dx > 0.1 and dy > 0.1:
            self.box_created.emit(cx, cy, cz, dx, dy, dz, 0.0)
                
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # 2D painter to draw on the top
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(255, 255, 255)) # white text
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        
        # Get Matrices for Projection
        view_matrix = self.viewMatrix()
        w = self.width()
        h = self.height()
        
        proj_matrix = get_projection_matrix(
            w, 
            h, 
            self.opts.get('fov', 60), 
            self.opts.get('distance', 20)
        )
       
        # Viewport is (x, y, width, height)
        viewport = QRect(0, 0, w, h)
        
        for box in self.overlay_boxes:
            # Get center of the box
            cx, cy, cz = box.x, box.y, box.z + (box.dz / 2.0) # Top of box
            
            # project 3D world to a 2D screen
            obj_vec = QVector3D(cx, cy, cz)
            screen_pos = obj_vec.project(view_matrix, proj_matrix, viewport)
            
            # If z is between 0 and 1, it's inside the frustum depth-wise
            if 0.0 <= screen_pos.z() <= 1.0:
                # We must invert Y: screen_y = height - projected_y
                screen_x = screen_pos.x()
                screen_y = h - screen_pos.y()
                
                # Draw the ID
                label_text = f"{box.track_id}: {box.label}"
                painter.drawText(int(screen_x), int(screen_y) - 10, label_text)
                
        painter.end()      
        
class LidarVisualizer(BasePluginWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.current_boxes = []
        self.box_items = []         # Visual Items (Lines)
        self.debug_items = []
        self.label_color_map = {}
        self.current_points = None
        self.current_metadata = {} # Cache for colors
        self.show_gt_boxes = False
        self.gt_box_items = [] # For GT boxes (Visual only)
        self.gt_box_cache = [] # List[BoundingBox3D]
        
    def set_label_colors(self, label_config: List[Dict]):
        """
        Populate the color lookup dictionary.
        Format: {'name': (r, g, b, a)} normalized to 0.0-1.0
        """
        self.label_color_map = {}
        for item in label_config:
            name = item["name"]
            rgb = item["color"]
            # Convert [255, 0, 0] -> (1.0, 0.0, 0.0, 1.0)
            self.label_color_map[name] = (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0, 1.0)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view_widget = CustomGLWidget()
        
        toolbar = QHBoxLayout()
        
        # COMPARISON CHECKBOX
        self.chk_compare = QCheckBox("Compare Ground Truth")
        self.chk_compare.setStyleSheet("color: #FF00FF; font-weight: bold;") # Magenta Text
        self.chk_compare.toggled.connect(self.toggle_comparison)
        toolbar.addWidget(self.chk_compare)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.view_widget.opts["distance"] = 40
        self.view_widget.setCameraPosition(elevation=30, azimuth=-90)
        
        # grid
        grid = gl.GLGridItem()
        grid.setSize(60, 60, 1)
        self.view_widget.addItem(grid)
        
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('additive')
        self.view_widget.addItem(self.scatter)

        layout.addWidget(self.view_widget)

    def on_frame_update(self, data: FrameData) -> None:
        if data.point_cloud is not None:
            # Safety Slice for (N, 4) -> (N, 3)
            self.current_points = data.point_cloud  # (N, 3)
            self.current_metadata = data.metadata # Save for GT access
            
            # Cache GT Boxes if available
            self.gt_boxes_cache = data.metadata.get("gt_boxes", [])
            
            # Enable checkbox only if GT exists
            self.chk_compare.setEnabled(len(self.gt_boxes_cache) > 0)
            if not self.gt_boxes_cache:
                self.chk_compare.setChecked(False)
            
            # 1. Restore Original Coloring (Height Gradient)
            self._draw_points_gradient()
            
            # 2. Redraw GT overlays if active
            if self.show_gt_boxes:
                self._draw_gt_overlays()
                
            # Initial Draw
            self.update_boxes(self.view_widget.overlay_boxes)
                
    def _draw_points_gradient(self):
        if self.current_points is None or len(self.current_points) == 0: 
            return
            
        pts = self.current_points[:, :3]
        z = pts[:, 2]
        
        # Color Logic: Blue (-2m) -> Cyan (0m) -> White (+3m)
        norm = np.clip((z + 2.0) / 5.0, 0, 1)
        
        colors = np.zeros((len(pts), 4))
        colors[:, 0] = 0.0              
        colors[:, 1] = norm * 0.9       
        colors[:, 2] = 0.6 + norm*0.4   
        
        # FIX: Force Alpha to 1.0 (Fully Opaque) for visibility check
        colors[:, 3] = 1.0              
        
        # Use pxMode=True for performant dots
        self.scatter.setData(pos=pts, color=colors, size=3, pxMode=True)
            
    def _update_point_cloud(self):
        if self.current_points is None: 
            return
        
        points_xyz = self.current_points[:, :3] # Slice XYZ
        num_pts = len(points_xyz)
        
        # Default: "NuScenes Blue" Style (Height Gradient)
        # Deep Blue (-2m) -> Cyan (0m) -> White (+3m)
        z = points_xyz[:, 2]
        
        # Create gradient 0.0 -> 1.0
        # Assuming ground is -1.8, roof is 0.0, trees/signs are +2.0
        norm_z = np.clip((z + 2.0) / 4.0, 0, 1) 
        
        colors = np.zeros((num_pts, 4), dtype=np.float32)
        colors[:, 0] = norm_z * 0.2        # Low Red
        colors[:, 1] = norm_z * 0.8        # High Green (Cyan-ish)
        colors[:, 2] = 0.8 + (norm_z * 0.2)# High Blue
        colors[:, 3] = 0.8                 # Alpha (Slightly transparent)

        # Override with GT if active
        if self.show_gt:
            gt = self.current_metadata.get('gt_colors')
            if gt is not None and len(gt) == num_pts:
                colors = gt
        
        self.scatter.setData(pos=points_xyz, color=colors, size=2, pxMode=True)
        
    def update_boxes(self, user_boxes):
        """Draws USER boxes (Green/Yellow)."""
        self.view_widget.overlay_boxes = user_boxes
        
        # 1. Update Box Wireframes (User)
        for item in self.box_items:
            self.view_widget.removeItem(item)
        self.box_items.clear()
        
        for box in user_boxes:
            self._draw_box_item(box, is_gt=False)

        # 2. Update Box Wireframes (GT)
        if self.show_gt_boxes:
            # Clear old GT items first if needed (usually handled in toggle)
            pass 
        else:
             # If GT is off, ensure no GT items
             for item in self.gt_box_items:
                 self.view_widget.removeItem(item)
             self.gt_box_items.clear()

        # 3. REPAINT POINTS (The "Trigger" you asked for)
        if self.current_points is not None:
            pts = self.current_points[:, :3]
            z = pts[:, 2]
            
            # Base Color (Blue Gradient)
            norm = np.clip((z + 2.0) / 5.0, 0, 1)
            colors = np.zeros((len(pts), 4))
            colors[:, 0] = 0.0
            colors[:, 1] = norm * 0.9       # Cyan
            colors[:, 2] = 0.6 + norm*0.4   # Blue
            colors[:, 3] = 1.0              # Opaque
            
            # A. Highlight USER Boxes (Green/Yellow)
            for box in user_boxes:
                target_color = self.label_color_map.get(box.label, (0.0, 1.0, 0.0, 1.0))
                if box.selected: 
                    target_color = (1.0, 1.0, 0.0, 1.0)
                
                indices = GeometryUtils.get_points_in_box(self.current_points, box)
                if len(indices) > 0:
                    colors[indices] = target_color

            # B. Highlight GT Boxes (Magenta) - IF ENABLED
            if self.show_gt_boxes:
                for gt_box in self.gt_boxes_cache:
                    indices = GeometryUtils.get_points_in_box(self.current_points, gt_box)
                    if len(indices) > 0:
                        # Bright Magenta for points inside GT
                        colors[indices] = (1.0, 0.0, 1.0, 1.0) 

            self.scatter.setData(pos=pts, color=colors, size=3, pxMode=True)
            
    def _draw_gt_overlays(self):
        """Draws GT boxes (Magenta/Purple)."""
        # Clear old GT items
        for item in self.gt_box_items:
            self.view_widget.removeItem(item)
        self.gt_box_items.clear()
        
        for box in self.gt_boxes_cache:
            self._draw_box_item(box, is_gt=True)
    
    def _draw_box_item(self, box, is_gt=False):
        corners = box.get_corners()
        lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        pts = []
        for s, e in lines:
            pts.append(corners[s])
            pts.append(corners[e])
            
        # Color Logic
        if is_gt:
            color = (1.0, 0.0, 1.0, 0.8) # Magenta for Ground Truth
            width = 1
        else:
            color = box.color # Green/Yellow (Selected)
            width = 2
            
        item = gl.GLLinePlotItem(pos=np.array(pts), mode='lines', color=color, width=width, antialias=True)
        self.view_widget.addItem(item)
        # trigger a repaint to draw the next text
        # self.view_widget.update()
        
        if is_gt:
            self.gt_box_items.append(item)
        else:
            self.box_items.append(item)

    def toggle_comparison(self, checked):
        self.show_gt_boxes = checked
       
        # 1. Handle Wireframes
        if checked:
            # Clear any old ones first
            for item in self.gt_box_items:
                self.view_widget.removeItem(item)
            self.gt_box_items.clear()
            
            # Draw new ones
            for box in self.gt_boxes_cache:
                self._draw_box_item(box, is_gt=True)
        else:
            # Remove all
            for item in self.gt_box_items:
                self.view_widget.removeItem(item)
            self.gt_box_items.clear()
            
        # 2. Handle Point Repaint
        self.update_boxes(self.view_widget.overlay_boxes)
            
    def _draw_points_default(self):
        """Standard Z-height gradient."""
        if self.current_points is None: 
            return
        
        z = self.current_points[:, 2]
        colors = np.ones((len(z), 4))
        colors[:, 0] = np.clip((z + 2.0) / 5.0, 0, 1) 
        colors[:, 1] = 0.5
        self.scatter.setData(pos=self.current_points, color=colors, size=2)

    def reset(self):
        self.scatter.setData(pos=np.zeros((0, 3)))
        
    def get_box_color(self, box):
        if box.selected:
            return (1, 1, 0, 1) # Yellow