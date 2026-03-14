from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGraphicsTextItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsPixmapItem,
)
from PyQt6.QtGui import QPixmap, QPen, QColor, QPainter
from PyQt6.QtCore import Qt, QRectF


class DrawableGraphicsView(QGraphicsView):
    """A custom GraphicsView that allows zooming and drawing a 2D bounding box."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.current_rect_item = None
        self.start_pt = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene = self.scene()
            if scene is None:
                return  # Safety check to satisfy the type-checker

            # Map the screen click to the actual image coordinates
            self.start_pt = self.mapToScene(event.pos())

            # Clear the old box if we are redrawing
            if self.current_rect_item:
                scene.removeItem(self.current_rect_item)

            self.current_rect_item = QGraphicsRectItem(
                QRectF(self.start_pt, self.start_pt)
            )
            self.current_rect_item.setPen(
                QPen(QColor(0, 255, 0), 2)
            )  # Bright Green Box
            scene.addItem(self.current_rect_item)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start_pt and self.current_rect_item:
            current_pt = self.mapToScene(event.pos())
            rect = QRectF(self.start_pt, current_pt).normalized()
            self.current_rect_item.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pt = None

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Scroll to zoom in and out."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)


class CameraPopOutModal(QDialog):
    """The pop-out window for 2D Image Annotation."""

    def __init__(
        self, pixmap: QPixmap, cam_name: str, existing_boxes=None, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"2D Annotation Modal - {cam_name}")
        self.resize(1280, 720)  # Start with a comfortably large window

        # Setup Scene and View
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)

        if existing_boxes:
            for box_data in existing_boxes:
                # Draw the rectangle
                rect = QRectF(
                    box_data["x"], box_data["y"], box_data["w"], box_data["h"]
                )
                rect_item = QGraphicsRectItem(rect)
                pen = QPen(box_data["color"], 3)
                if box_data.get("is_override"):
                    pen.setStyle(Qt.PenStyle.SolidLine)
                rect_item.setPen(pen)
                self.scene.addItem(rect_item)

                # Draw the Text Label with a semi-transparent background
                text_item = QGraphicsTextItem()
                html_str = f"<div style='background-color: rgba(0,0,0,0.6); color: white; padding: 2px;'><b>{box_data['label']}</b></div>"
                text_item.setHtml(html_str)
                text_item.setPos(
                    box_data["x"], box_data["y"] - 30
                )  # Position just above the box
                self.scene.addItem(text_item)

        self.view = DrawableGraphicsView(self.scene)

        # Remove fitInView so the image defaults to a massive 1:1 pixel scale.
        self.view.resetTransform()

        # Setup UI Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_save = QPushButton("Confirm 2D Box")
        self.btn_save.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold;"
        )

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        # Connect signals
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self.accept)

    def get_bounding_box(self) -> QRectF:
        """Returns the [x, y, width, height] of the drawn box in raw image coordinates."""
        if self.view.current_rect_item:
            return self.view.current_rect_item.rect()
        return None
