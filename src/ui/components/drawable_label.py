from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent, QFont, QFontMetrics
import logging
from src.core.geometry import GeometryUtils
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DrawableLabel(QLabel):
    """
    A QLabel that allows the user to click and drag to draw a box.
    Emits the box coordinates normalized to the ORIGINAL image size.
    """

    # Signal: (x, y, w, h) in original image coordinates
    selection_finished = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # State
        self.start_point: Optional[QPoint] = None
        self.current_rect: Optional[QRect] = None
        self.is_drawing = False

        # orginal resolution (set when image is updated)
        self.orig_width = 1
        self.orig_height = 1

        # List of [x, y, w, h] in ORIGINAL coordinates
        self.static_rects = []
        
        # Live Projection Data
        self.current_boxes_3d = []
        self.intrinsic = None
        self.extrinsic = None
        self.camera_id = None
        
        # Default Map (Green Fallback)
        self.label_color_map = {}

    def set_original_resolution(self, w: int, h: int) -> None:
        self.orig_width = w
        self.orig_height = h

    def set_projection_data(self, boxes, intrinsic, extrinsic):
        """
        Updates the 3D data used for live projection.
        """
        self.current_boxes_3d = boxes
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.update()  # Trigger repaint
    
    def set_static_rects(self, rects_data):
        """
        Receives list of dicts: [{'rect': [x,y,w,h], 'id': 1, 'label': 'person'}, ...]
        """
        self.static_rects = rects_data
        self.update()
        
    def set_camera_id(self, cam_id: str):
        self.camera_id = cam_id
        
    def set_label_colors(self, label_config: List[Dict]):
        """
        Populate the color lookup dictionary.
        Format: {'name': QColor(r, g, b)}
        """
        self.label_color_map = {}
        for item in label_config:
            name = item["name"]
            rgb = item["color"]
            # Store as QColor for fast drawing
            self.label_color_map[name] = QColor(rgb[0], rgb[1], rgb[2])
            
    def get_view_params(self):
        """
        Calculates the scale and offsets to center the image while preserving aspect ratio.
        Returns: (scale, offset_x, offset_y)
        """
        if self.orig_width == 0 or self.orig_height == 0:
            return 1.0, 0, 0
            
        widget_w = self.width()
        widget_h = self.height()
        
        scale_x = widget_w / self.orig_width
        scale_y = widget_h / self.orig_height
        
        # Use the smaller scale to fit the image entirely (Letterbox)
        scale = min(scale_x, scale_y)
        
        # Calculate centering offsets
        new_w = int(self.orig_width * scale)
        new_h = int(self.orig_height * scale)
        
        offset_x = (widget_w - new_w) // 2
        offset_y = (widget_h - new_h) // 2
        
        return scale, offset_x, offset_y

    def _clamp_to_image_area(self, point: QPoint) -> QPoint:
        """Clamp a widget coordinate point to the displayed image area."""
        scale, off_x, off_y = self.get_view_params()
        target_w = int(self.orig_width * scale)
        target_h = int(self.orig_height * scale)
        
        # Image bounds in widget coordinates
        img_left = off_x
        img_right = off_x + target_w
        img_top = off_y
        img_bottom = off_y + target_h
        
        # Clamp point to image area
        clamped_x = max(img_left, min(point.x(), img_right))
        clamped_y = max(img_top, min(point.y(), img_bottom))
        
        return QPoint(clamped_x, clamped_y)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scale, off_x, off_y = self.get_view_params()
            target_w = int(self.orig_width * scale)
            target_h = int(self.orig_height * scale)
            img_rect = QRect(off_x, off_y, target_w, target_h)
            
            # Only start drawing if click is within image area
            if img_rect.contains(event.position().toPoint()):
                self.is_drawing = True
                self.start_point = self._clamp_to_image_area(event.position().toPoint())
                self.current_rect = QRect(self.start_point, self.start_point)
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_drawing and self.start_point:
            current_pos = self._clamp_to_image_area(event.position().toPoint())
            self.current_rect = QRect(self.start_point, current_pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False

            # Get the drawn rect on screen
            screen_rect = self.current_rect

            # Convert to Original Image Coordinates
            if self.pixmap() and screen_rect is not None and not screen_rect.isEmpty():
                scale, off_x, off_y = self.get_view_params()
                
                # Convert Widget Coords -> Image Coords
                # formula: img_x = (screen_x - offset) / scale
                real_x = (screen_rect.x() - off_x) / scale
                real_y = (screen_rect.y() - off_y) / scale
                real_w = screen_rect.width() / scale
                real_h = screen_rect.height() / scale
                
                # Clamp to image bounds and adjust dimensions if needed
                real_x = max(0, min(real_x, self.orig_width - 1))
                real_y = max(0, min(real_y, self.orig_height - 1))
                
                # Ensure width and height don't extend beyond image bounds
                if real_x + real_w > self.orig_width:
                    real_w = self.orig_width - real_x
                if real_y + real_h > self.orig_height:
                    real_h = self.orig_height - real_y
                
                # Convert to integers
                real_x = int(real_x)
                real_y = int(real_y)
                real_w = max(1, int(real_w))  # Ensure at least 1 pixel
                real_h = max(1, int(real_h))  # Ensure at least 1 pixel
                
                # Emit the signal
                logger.info(
                    f"Image Box Drawn: {real_x}, {real_y}, {real_w}x{real_h}"
                )
                self.selection_finished.emit(real_x, real_y, real_w, real_h)

            # Clear visual box after release
            self.current_rect = None
            self.update()

    def paintEvent(self, event):
        # draw the image
        # super().paintEvent(event)
        
        # Do NOT call super().paintEvent(event) 
        # because that draws the stretched image if setScaledContents(True) is used.
        # We will draw the pixmap manually with correct aspect ratio.
        
        if not self.pixmap():
            return
        
        # If no calibration, we can't project
        if self.intrinsic is None or self.extrinsic is None:
            return
        
        painter = QPainter(self)
        
        # Draw Pixmap (Centered & Scaled)
        scale, off_x, off_y = self.get_view_params()
        
        target_w = int(self.orig_width * scale)
        target_h = int(self.orig_height * scale)
        
        # This draws the image exactly where our math expects it
        target_rect = QRect(off_x, off_y, target_w, target_h)
        painter.drawPixmap(target_rect, self.pixmap())

        # Setup Font
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # # Calculate Scale (Widget Size vs Original Image Size)
        # scale_x = self.width() / self.orig_width
        # scale_y = self.height() / self.orig_height

        for box in self.current_boxes_3d:
            rect_2d = None
            is_manual_override = False
            
            overrides = getattr(box, "visual_overrides", {})
            
            if self.camera_id and self.camera_id in overrides:
                rect_2d = overrides[self.camera_id]
                is_manual_override = True
            else:
                # Use standard 3D projection for all other cameras
                rect_2d = GeometryUtils.project_box_to_image(
                    box, self.extrinsic, self.intrinsic, (self.orig_height, self.orig_width)
                )
            
            if not rect_2d:
                continue
            
            rx, ry, rw, rh = rect_2d

            # Scale back to screen coords
            sx = int(rx * scale + off_x)
            sy = int(ry * scale + off_y)
            sw = int(rw * scale)
            sh = int(rh * scale)
            
            # Define Color
            if box.selected:
                color = QColor(255, 255, 0) # Yellow
            else:
                # Dynamic Lookup
                color = self.label_color_map.get(box.label, QColor(0, 255, 0))

            # Draw Box
            pen = QPen(color, 2)
            if is_manual_override:
                 # Visual cue: line implies "Manual Edit"
                 pen.setStyle(Qt.PenStyle.SolidLine)
                 
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(sx, sy, sw, sh)

            # Draw label
            label_text = f"{box.track_id}: {box.label}"
            
            # Draw text overlay
            text_w = fm.horizontalAdvance(label_text) + 10
            text_h = fm.height() + 4
            text_y = sy - text_h if sy - text_h > 0 else sy

            # Draw tiny background for text (so it's readable)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 150))  # Semi-transparent black
            painter.drawRect(sx, text_y, text_w, text_h)

            # Draw Text
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.drawText(sx + 5, text_y + fm.ascent() + 2, label_text)

        # Draw ACTIVE Rubberband (The one you are dragging) - Green
        if self.current_rect and self.is_drawing:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 50))
            painter.drawRect(self.current_rect)
