from PyQt6.QtGui import QPainter, QColor, QFont, QVector3D, QKeyEvent
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
        self.gt_boxes = []

        # Drawing State
        self.state = DrawState.IDLE
        self.base_points = []         # List of clicked [x, y, z] points
        self.current_mouse_pt = None  
                
        self.draw_ground_z = None   # Z of ground for current draw (set from first click)
        self.draw_height = 0      # Default height
        self.ground_z = 0       # Default ground plane for first ray cast (can be overridden by config)
        self._height_drag_start_y = None  # For mouse-drag height adjustment

        # Ghost Box (Visual Feedback while drawing)
        self.ghost_box = gl.GLBoxItem(color=(0, 255, 255, 255)) # Cyan
        self.ghost_box.setVisible(False)
        self.addItem(self.ghost_box)
        
        self.ghost_pts = gl.GLScatterPlotItem(color=(1, 1, 0, 1), size=6) # Yellow dots
        self.ghost_pts.setVisible(False)
        self.addItem(self.ghost_pts)
        
        # This helps confirm if your Lidar is rotated correctly.
        axis = gl.GLAxisItem()
        axis.setSize(3, 3, 3)
        self.addItem(axis)
        

    def _screen_to_ray_qt(self, mouse_x: int, mouse_y: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a world-space picking ray using Qt's unproject (avoids matrix convention issues).

        `QVector3D.project()` returns coordinates with Y-up, so we flip the incoming
        mouse Y (top-left) into window coordinates (bottom-left) before unproject.
        """
        rect = self.rect()
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return np.zeros(3), np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Normalized Device Coordinates [-1 to 1]
        ndc_x = (2.0 * mouse_x / w) - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y / h)

        # Extract EXACT matrices used for rendering
        proj = self.projectionMatrix(rect.getRect(), rect.getRect())
        view = self.viewMatrix()
        inv_vp, invertible = (proj * view).inverted()

        if not invertible:
            return np.zeros(3), np.array([0.0, 0.0, 1.0], dtype=np.float32)

        near_pt = inv_vp.map(QVector3D(ndc_x, ndc_y, -1.0))
        far_pt = inv_vp.map(QVector3D(ndc_x, ndc_y, 1.0))

        origin = np.array([near_pt.x(), near_pt.y(), near_pt.z()], dtype=np.float32)
        end = np.array([far_pt.x(), far_pt.y(), far_pt.z()], dtype=np.float32)
        
        direction = end - origin
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        return origin, direction
    
    def _calculate_obb(self):
        """Calculates Center, Dims, and Heading from 1 to 4 base points."""
        if len(self.base_points) == 0:
            return None
        
        p1 = self.base_points[0]
        p2 = self.current_mouse_pt if len(self.base_points) == 1 else self.base_points[1]
        if p2 is None: p2 = p1
            
        dx = max(0.01, np.linalg.norm(p2[:2] - p1[:2]))
        heading = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        
        # --- MAGNETIC AXIS SNAPPING ---
        # Snaps to perfectly straight 0, 90, 180, 270 degrees if within 4 degrees
        snap_rad = np.radians(4.0)
        for target in [0, np.pi/2, np.pi, -np.pi/2, -np.pi]:
            if abs(heading - target) < snap_rad:
                heading = target
                # Override p2 visually so the yellow dot snaps into perfect alignment
                p2 = p1.copy()
                p2[0] += dx * np.cos(heading)
                p2[1] += dx * np.sin(heading)
                break
                
        if len(self.base_points) == 1:
            cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            return cx, cy, self.draw_ground_z, dx, 0.01, heading
            
        p3 = self.base_points[2] if len(self.base_points) > 2 else self.current_mouse_pt
        if p3 is not None:
            v_perp = np.array([-np.sin(heading), np.cos(heading)])
            dy_vector = p3[:2] - p1[:2]
            dy = abs(np.dot(dy_vector, v_perp))
            if dy < 0.01: dy = 0.01
            
            sign = np.sign(np.dot(dy_vector, v_perp))
            if sign == 0: sign = 1
            
            midpoint = (p1 + p2) / 2.0
            center_xy = midpoint[:2] + v_perp * sign * (dy / 2.0)
        else:
            dy = 0.01
            center_xy = (p1[:2] + p2[:2]) / 2.0

        return center_xy[0], center_xy[1], self.draw_ground_z, dx, dy, heading
            
    def mousePressEvent(self, event):
        # using ctrl+left click start the process
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                logger.info(f"Ctrl+Click detected at {event.pos()}")
                origin, direction = self._screen_to_ray_qt(event.pos().x(), event.pos().y())
                
                if self.state == DrawState.IDLE:
                    hit = GeometryUtils.intersect_ray_plane(origin, direction, self.ground_z)
                    
                    if hit is not None:
                        logger.info("State changed to DRAGGING_BASE")
                        self.state = DrawState.DRAGGING_BASE
                        self.base_points = [hit.copy()]
                        self.draw_ground_z = float(hit[2])
                        self.draw_height = 0.01
                        self._update_ghost_box()
                        
                    else:
                        logger.info("Ray missed the ground plane!")
                        
                elif self.state == DrawState.DRAGGING_BASE:
                    hit = GeometryUtils.intersect_ray_plane(origin, direction, self.draw_ground_z)
                    if hit is not None:
                        # Append the magnetically snapped point if it's point 2
                        if len(self.base_points) == 1:
                            obb = self._calculate_obb()
                            if obb:
                                heading = obb[5]
                                dx = obb[3]
                                hit = self.base_points[0].copy()
                                hit[0] += dx * np.cos(heading)
                                hit[1] += dx * np.sin(heading)
                        
                        self.base_points.append(hit.copy())
                        if len(self.base_points) == 4:
                            self.state = DrawState.SETTING_HEIGHT
                            self.draw_height = 1.6  # Extrude visually
                        self._update_ghost_box()
                        
                elif self.state == DrawState.SETTING_HEIGHT:
                    # finish drawing
                    self._finalize_drawing()
                    self._reset_draw_state()
                    # self.ghost_box.setVisible(False)
                event.accept()
                return
                
        else:
            # No Ctrl: allow height drag in SETTING_HEIGHT (left press starts drag)
            if self.state == DrawState.SETTING_HEIGHT and event.button() == Qt.MouseButton.LeftButton:
                self._height_drag_start_y = event.pos().y()
                event.accept()
                return
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        if self.state == DrawState.DRAGGING_BASE:
            plane_z = self.draw_ground_z if self.draw_ground_z is not None else self.ground_z
            origin, direction = self._screen_to_ray_qt(event.pos().x(), event.pos().y())
            hit = GeometryUtils.intersect_ray_plane(origin, direction, plane_z)
            if hit is not None:
                self.current_mouse_pt = hit
                self._update_ghost_box()
                
        elif self.state == DrawState.SETTING_HEIGHT:
            # Mouse drag to adjust height: vertical delta -> height change
            if self._height_drag_start_y is not None:
                delta_y = self._height_drag_start_y - event.pos().y()  # up = taller
                self.draw_height = max(0.2, self.draw_height + delta_y * 0.02)
                self._height_drag_start_y = event.pos().y()
                self._update_ghost_box()
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.state == DrawState.SETTING_HEIGHT:
            self._height_drag_start_y = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self.state == DrawState.SETTING_HEIGHT:
            delta = event.angleDelta().y()
            self.draw_height = max(0.2, self.draw_height + (delta / 120.0) * 0.2)
            self._update_ghost_box()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            if self.state != DrawState.IDLE:
                self._reset_draw_state()
                self.ghost_box.setVisible(False)
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)
    
    def _reset_draw_state(self):
        """Clear drawing state after confirm or cancel."""
        self.state = DrawState.IDLE
        self.draw_start_pt = None
        self.draw_end_pt = None
        self.draw_ground_z = None
        self.draw_height = None
        self._height_drag_start_y = None
        
        # 1. Force the bounding box to disappear and collapse to zero
        self.ghost_box.setVisible(False)
        # self.ghost_box.setSize(0.001, 0.001, 0.001) 
        
        # 2. Force the yellow scatter points to clear by passing an empty array
        self.ghost_pts.setVisible(False)
        # self.ghost_pts.setData(pos=np.empty((0, 3)))
        
        # 3. Trigger a full widget repaint to flush the OpenGL buffer
        self.update()

    def _update_ghost_box(self):
        if not self.base_points:
            return
            
        # Draw the points
        pts_to_draw = list(self.base_points)
        if self.current_mouse_pt is not None and len(self.base_points) < 4:
            pts_to_draw.append(self.current_mouse_pt)
        self.ghost_pts.setData(pos=np.array(pts_to_draw))
        self.ghost_pts.setVisible(True)

        # Draw the OBB Wireframe
        obb = self._calculate_obb()
        if not obb: return
        cx, cy, cz, dx, dy, heading = obb
        
        dz = max(0.01, self.draw_height)
        cz_center = cz + (dz / 2.0)
        
        self.ghost_box.setSize(dx, dy, dz)
        self.ghost_box.resetTransform()
        self.ghost_box.translate(-dx/2.0, -dy/2.0, -dz/2.0) # Local Center
        self.ghost_box.rotate(np.degrees(heading), 0, 0, 1) # Apply Heading
        self.ghost_box.translate(cx, cy, cz_center)         # Move to World Pos
        self.ghost_box.setVisible(True)
        self.update()
    
    def _finalize_drawing(self):
        obb = self._calculate_obb()
        if not obb: 
            return
        cx, cy, cz, dx, dy, heading = obb
        dz = max(0.01, self.draw_height)
        cz_center = cz + (dz / 2.0)
        
        self.box_created.emit(cx, cy, cz_center, dx, dy, dz, heading)
                
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # 2D painter to draw on the top
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))

        # Draw status hint when manually drawing a box
        if self.state == DrawState.DRAGGING_BASE:
            pts = len(self.base_points)
            if pts == 1: painter.drawText(10, 24, "Click 2: Set Length & Heading")
            elif pts == 2: painter.drawText(10, 24, "Click 3: Set Width")
            elif pts == 3: painter.drawText(10, 24, "Click 4: Confirm Base")
        elif self.state == DrawState.SETTING_HEIGHT:
            painter.drawText(10, 24, "Ctrl+Click: Confirm Box | Scroll/Drag: Adjust Height")
        
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
                
        painter.setPen(QColor(255, 0, 255)) 
        
        for box in self.gt_boxes:
            cx, cy, cz = box.x, box.y, box.z + (box.dz / 2.0)
            obj_vec = QVector3D(cx, cy, cz)
            screen_pos = obj_vec.project(view_matrix, proj_matrix, viewport)
            
            if 0.0 <= screen_pos.z() <= 1.0:
                screen_x = screen_pos.x()
                screen_y = h - screen_pos.y()
                # GT doesn't usually have a track_id in NuScenes mini, or it's very long
                # We show "GT: <Label>"
                label_text = f"GT: {box.label}"
                painter.drawText(int(screen_x), int(screen_y) - 10, label_text)
                
        painter.end()           
        
class LidarVisualizer(BasePluginWidget):
    box_selected_3d = pyqtSignal(int)
    
    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        # Configuration for ground plane estimation and defaults
        self._cfg = cfg
        self.ground_percentile = getattr(cfg, "ground_percentile", 0.5) if cfg is not None else 0.5
        self.ground_bias = getattr(cfg, "ground_bias", 0.05) if cfg is not None else 0.05
        self.default_ground_z = getattr(cfg, "default_ground_z", -1.5) if cfg is not None else -1.5
        
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
        
        # Intercept the mouse release event to detect clicks
        self._original_mouse_release = self.view_widget.mouseReleaseEvent
        self.view_widget.mouseReleaseEvent = self._on_gl_mouse_release
        
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
        
        # Save a reference to the boxes so we can test against them later
        self.current_boxes = user_boxes
        
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
        self.view_widget.gt_boxes = self.gt_boxes_cache
        
        for item in self.gt_box_items:
            self.view_widget.removeItem(item)
        self.gt_box_items.clear()
        
        for box in self.gt_boxes_cache:
            self._draw_box_item(box, is_gt=True)
            
        self.view_widget.update()
    
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
       
        # Handle Wireframes
        if checked:
            # Pass data to widget so paintEvent can draw text labels
            self.view_widget.gt_boxes = self.gt_boxes_cache
            
            # Clear any old ones first
            for item in self.gt_box_items:
                self.view_widget.removeItem(item)
            self.gt_box_items.clear()
            
            # Draw new ones
            for box in self.gt_boxes_cache:
                self._draw_box_item(box, is_gt=True)
        else:
            self.view_widget.gt_boxes = []
            
            # Remove all
            for item in self.gt_box_items:
                self.view_widget.removeItem(item)
            self.gt_box_items.clear()
         
        # Force Repaint for Text
        self.view_widget.update() 
            
        # Handle Point Repaint
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
        
    def _on_gl_mouse_release(self, ev):
        # Ensure the camera pan/rotate still works natively
        self._original_mouse_release(ev)
        
        # Only trigger selection on Left Click (Modify to Shift+Click if you prefer)
        if ev.button() != Qt.MouseButton.LeftButton:
            return
            
        # Get 2D Pixel Coordinates
        pos = ev.pos()
        rect = self.view_widget.rect()
        
        # Convert to Normalized Device Coordinates (NDC) [-1 to 1]
        ndc_x = (2.0 * pos.x() / rect.width()) - 1.0
        ndc_y = 1.0 - (2.0 * pos.y() / rect.height())
        
        # Get the Inverse View-Projection Matrix to unproject the 2D point
        proj = self.view_widget.projectionMatrix(rect.getRect(), rect.getRect())
        view = self.view_widget.viewMatrix()
        vp_matrix = proj * view
        inv_vp, invertible = vp_matrix.inverted()
        
        if not invertible:
            return
            
        # Unproject Near (z=-1) and Far (z=1) points to create the 3D Ray
        near_pt = inv_vp.map(QVector3D(ndc_x, ndc_y, -1.0))
        far_pt = inv_vp.map(QVector3D(ndc_x, ndc_y, 1.0))
        
        ray_origin = np.array([near_pt.x(), near_pt.y(), near_pt.z()])
        ray_far = np.array([far_pt.x(), far_pt.y(), far_pt.z()])
        
        ray_dir = ray_far - ray_origin
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-6: 
            return
        ray_dir /= norm  # Normalize direction vector
        
        # Raycast against all current bounding boxes
        closest_hit_id = -1
        min_dist = np.inf
        
        for box in self.current_boxes:
            hit, dist = GeometryUtils.ray_intersects_obb(ray_origin, ray_dir, box)
            if hit and dist < min_dist:
                min_dist = dist
                closest_hit_id = box.track_id
                
        # Emit the selected ID to the main window
        if closest_hit_id != -1:
            self.box_selected_3d.emit(closest_hit_id)