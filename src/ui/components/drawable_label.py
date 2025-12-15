from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QMouseEvent


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

    def set_original_resolution(self, w: int, h: int) -> None:
        self.orig_width = w
        self.orig_height = h
        
    def set_static_rects(self, rects_list):
        self.static_rects = rects_list
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
                # calulate scale factors
                disp_w = self.width()
                disp_h = self.height()

                # Note: QLabel usually centers the image if scaled.
                # For simplicity, we assume the pixmap fills the label (ScaledContents=True)
                # or we calculate the offset.

                scale_x = self.orig_width / disp_w
                scale_y = self.orig_height / disp_h

                real_x = int(screen_rect.x() * scale_x)
                real_y = int(screen_rect.y() * scale_y)
                real_w = int(screen_rect.width() * scale_x)
                real_h = int(screen_rect.height() * scale_y)

                # Emit the signal
                print(
                    f"[STATUS]: Image Box Drawn: {real_x}, {real_y}, {real_w}x{real_h}"
                )
                self.selection_finished.emit(real_x, real_y, real_w, real_h)

            # Clear visual box after release
            self.current_rect = None
            self.update()

    def paintEvent(self, event):
        # draw the image
        super().paintEvent(event)
        painter = QPainter(self)

        # Calculate Scale
        scale_x = self.width() / self.orig_width
        scale_y = self.height() / self.orig_height
        
        # Draw STATIC Boxes (The ones saved) - Cyan/Blue
        pen_static = QPen(QColor(0, 255, 255), 2) # Cyan
        painter.setPen(pen_static)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        for (rx, ry, rw, rh) in self.static_rects:
            # Scale back to screen coords
            sx = int(rx * scale_x)
            sy = int(ry * scale_y)
            sw = int(rw * scale_x)
            sh = int(rh * scale_y)
            painter.drawRect(sx, sy, sw, sh)

        # Draw ACTIVE Rubberband (The one you are dragging) - Green
        if self.current_rect and self.is_drawing:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 50))
            painter.drawRect(self.current_rect)
