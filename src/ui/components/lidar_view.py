from PyQt6.QtGui import QPainter, QColor, QFont, QVector3D
from pyqtgraph.Qt.QtGui import QMatrix4x4
from PyQt6.QtCore import Qt, QRect
import pyqtgraph.opengl as gl
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout
from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData
from src.core.objects import BoundingBox3D

class CustomGLWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.overlay_boxes = []
        
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
        fov = self.opts.get('fov', 60) # Default to 60 if missing
        near_clip = 0.01
        far_clip = 10000.0
        
        proj_matrix = QMatrix4x4()
        
        # Standard perspective projection: fov (deg), aspect ratio, near, far
        proj_matrix.perspective(fov, w / h, near_clip, far_clip)
       
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
        self.box_items = []
        self.debug_items = []
        self.current_points = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view_widget = CustomGLWidget()
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
            self.current_points = data.point_cloud  # (N, 3)

            # Color Map Logic (Height Based)
            z = self.current_points[:, 2]
            colors = np.ones((self.current_points.shape[0], 4))
            colors[:, 0] = np.clip((z + 2) / 5, 0, 1)  # R
            colors[:, 1] = 0.5  # G

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
                if box.point_indices is not None:
                    # Color these points RED
                    lbl = box.label.strip() if box.label else "unknown"
                    target_color = class_colors.get(lbl, default_color)

                    # Apply to the specific indices
                    colors[box.point_indices] = target_color

            # Update the scatter plot
            self.scatter.setData(pos=self.current_points, color=colors, size=2)

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
                pos=pts, mode="lines", color=box.color, width=2, antialias=True
            )

            self.view_widget.addItem(line_item)
            self.box_items.append(line_item)
            
        # trigger a repaint to draw the next text
        self.view_widget.update()

    def reset(self):
        self.scatter.setData(pos=np.zeros((0, 3)))

    def draw_debug_lines(self, lines_list):
        """Draws persistent debug rays."""
        for item in self.debug_items:
            self.view_widget.removeItem(item)
        self.debug_items.clear()

        if not lines_list:
            return

        pts = []
        for line in lines_list:
            pts.append(line[0])  # Start (Camera Origin)
            pts.append(line[1])  # End (Frustum corner)

        pts_arr = np.array(pts)

        print(f"DEBUG: Drawing {len(lines_list)} lines.")
        print(f"DEBUG: Start Point (Cam): {pts_arr[0]}")
        print(f"DEBUG: End Point (Ray): {pts_arr[1]}")

        line_item = gl.GLLinePlotItem(
            pos=pts_arr,
            mode="lines",
            color=(1, 0, 0, 1),  # Bright Red
            width=3,  # Thicker lines
            antialias=True,
        )
        self.view_widget.addItem(line_item)
        self.debug_items.append(line_item)
