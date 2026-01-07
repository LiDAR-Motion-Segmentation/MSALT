from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent, QFont, QFontMetrics
import logging
from src.core.geometry import GeometryUtils

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

        # State
        self.start_point: QPoint = None
        self.current_rect: QRect = None
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.start_point = event.position().toPoint()
            self.current_rect = QRect(self.start_point, self.start_point)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_drawing and self.start_point:
            current_pos = event.position().toPoint()
            self.current_rect = QRect(self.start_point, current_pos).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False

            # Get the drawn rect on screen
            screen_rect = self.current_rect

            # Convert to Original Image Coordinates
            if self.pixmap() and not screen_rect.isEmpty():
                # Calculate scale to map back to original image resolution
                disp_w = self.width()
                disp_h = self.height()

                # For simplicity, we assume the pixmap fills the label (ScaledContents=True)
                # or we calculate the offset.

                scale_x = self.orig_width / disp_w
                scale_y = self.orig_height / disp_h

                real_x = int(screen_rect.x() * scale_x)
                real_y = int(screen_rect.y() * scale_y)
                real_w = int(screen_rect.width() * scale_x)
                real_h = int(screen_rect.height() * scale_y)

                # Emit the signal
                logger.info(
                    f"Image Box Drawn: {real_x}, {real_y}, {real_w}x{real_h}"
                )
                self.selection_finished.emit(real_x, real_y, real_w, real_h)

            # Clear visual box after release
            self.current_rect = None
            self.update()
            
    def set_camera_id(self, cam_id: str):
        self.camera_id = cam_id

    def paintEvent(self, event):
        # draw the image
        super().paintEvent(event)
        
        # If no calibration, we can't project
        if self.intrinsic is None or self.extrinsic is None:
            return
        
        painter = QPainter(self)

        # Setup Font
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Calculate Scale (Widget Size vs Original Image Size)
        scale_x = self.width() / self.orig_width
        scale_y = self.height() / self.orig_height

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
            sx = int(rx * scale_x)
            sy = int(ry * scale_y)
            sw = int(rw * scale_x)
            sh = int(rh * scale_y)
            
            # Define Color
            if box.selected:
                color = QColor(255, 255, 0) # Yellow
            elif box.label == "moving_people":
                color = QColor(255, 0, 0)   # Red
            elif box.label == "static_people":
                color = QColor(0, 255, 0)   # Green
            else:
                color = QColor(0, 150, 255) # Blue

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
