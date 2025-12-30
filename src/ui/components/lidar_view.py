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
                        event.pos().x(), event.pos().y(),
                        self.width(), self.height(),
                        vm, pm 
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
                event.pos().x(), event.pos().y(),
                self.width(), self.height(),
                vm, pm
            )
            
            hit = GeometryUtils.intersect_ray_plane(origin, direction, self.ground_z)
            
            if hit is not None:
                self.draw_end_pt = hit
                self._update_ghost_box()
                
        elif self.state == DrawState.SETTING_HEIGHT:
            # Update Height (Visual only)
            # We map vertical mouse movement to Z height
            # Simple heuristic: 100 pixels = 2 meters
            # Ideally we track delta from mouseRelease, but absolute Y works for now
            pass
            # (Refinement: We can implement pixel-delta logic here if needed)
        
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
    def __init__(self):
        super().__init__(title="LiDAR 3D View")
        self._setup_ui()
        self.current_boxes = []
        self.box_items = []         # Visual Items (Lines)
        self.debug_items = []
        self.current_points = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view_widget = CustomGLWidget()
        self.view_widget.opts["distance"] = 20
        self.view_widget.setWindowTitle("LiDAR Viewer")

        # grid
        grid = gl.GLGridItem()
        grid.setSize(x=50, y=50, z=1)
        self.view_widget.addItem(grid)

        # Scatter Plot (The Point Cloud)
        self.scatter = gl.GLScatterPlotItem()
        self.view_widget.addItem(self.scatter)

        layout.addWidget(self.view_widget)

    def on_frame_update(self, data: FrameData) -> None:
        if data.point_cloud is not None:
            self.current_points = data.point_cloud  # (N, 3)
            
            if not self.view_widget.overlay_boxes:
                self._draw_points_default()
            
    def _draw_points_default(self):
        """Standard Z-height gradient."""
        if self.current_points is None: 
            return
        
        z = self.current_points[:, 2]
        colors = np.ones((len(z), 4))
        colors[:, 0] = np.clip((z + 2.0) / 5.0, 0, 1) 
        colors[:, 1] = 0.5
        self.scatter.setData(pos=self.current_points, color=colors, size=2)

    def update_boxes(self, boxes: list[BoundingBox3D]):
        self.view_widget.overlay_boxes = boxes
        
        # clear old boxes
        for item in self.box_items:
            self.view_widget.removeItem(item)
        self.box_items.clear()

        # Re-Color Point Cloud (Highlight Selected Points)
        if self.current_points is not None:
            # Reset to default colors first
            points = self.current_points
            z = self.current_points[:, 2]
            colors = np.ones((len(z), 4))
            colors[:, 0] = np.clip((z + 2) / 5, 0, 1)
            colors[:, 1] = 0.5

            class_colors = {
                "moving_people": [1.0, 0.0, 0.0, 1.0],  # Red
                "static_people": [1.0, 1.0, 0.0, 1.0],  # Yellow
                "static_car": [0.0, 0.5, 1.0, 1.0],  # Sky Blue
                "cyclist": [1.0, 0.5, 0.0, 1.0],  # Orange
                "noise": [0.5, 0.0, 0.5, 1.0],  # Purple
            }
            default_color = [0.0, 1.0, 0.0, 1.0]  # Green

            for box in boxes:
                # Color these points RED
                lbl = box.label.strip() if box.label else "unknown"
                target_color = class_colors.get(lbl, default_color)

                indices = GeometryUtils.get_points_in_box(points, box)
                    
                if len(indices) > 0:
                    colors[indices] = target_color
                    # box.color = target_color

            # Update the scatter plot
            self.scatter.setData(pos=points, color=colors, size=2)

        # Draw new box lines
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

            # GLLinePlotItem usually takes a list of points in sequence for 'lines' mode
            # Construct line pairs manually for GLLinePlotItem to be safe
            pts = []
            for start, end in lines_indices:
                pts.append(corners[start])
                pts.append(corners[end])

            pts = np.array(pts)

            # create line item
            line_item = gl.GLLinePlotItem(
                pos=pts, 
                mode="lines", 
                color=box.color, 
                width=2, 
                antialias=True
            )
            self.view_widget.addItem(line_item)
            self.box_items.append(line_item)
            
        # trigger a repaint to draw the next text
        self.view_widget.update()

    def reset(self):
        self.scatter.setData(pos=np.zeros((0, 3)))
