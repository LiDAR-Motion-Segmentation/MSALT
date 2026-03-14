from typing import List, Dict
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QScrollArea,
    QDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData
from src.ui.components.drawable_label import DrawableLabel
from src.ui.components.camera_modal import CameraPopOutModal
from src.core.objects import BoundingBox3D
import logging

logger = logging.getLogger(__name__)


class CameraStripWidget(BasePluginWidget):
    """
    Manages the row of camera views.
    """

    # (CameraID, x, y, w, h, is_override)
    box_drawn = pyqtSignal(str, int, int, int, int, bool)
    pixel_hovered = pyqtSignal(str, float, float)  # cam_id, u, v

    def __init__(self, camera_ids: List[str]):
        super().__init__(title="camera strip")
        self.camera_ids = camera_ids
        self.image_labels: Dict[str, DrawableLabel] = {}
        self.label_config: List = []
        self._setup_ui()

    def set_label_config(self, label_config: List[Dict]):
        self.label_config = label_config

        # self.image_labels is a dict: {'CAM_FRONT': DrawableLabel, ...}
        if hasattr(self, "image_labels"):
            for cam_id, label_widget in self.image_labels.items():
                label_widget.set_label_colors(label_config)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.strip_layout = QHBoxLayout(container)

        for cam_id in self.camera_ids:
            # Container for 1 Camera (Label + Image)
            cam_box = QWidget()
            v_layout = QVBoxLayout(cam_box)
            v_layout.setContentsMargins(2, 2, 2, 2)

            # Title
            lbl_title = QLabel(cam_id)
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Image Placeholder
            lbl_img = DrawableLabel()
            if hasattr(self, "label_config"):
                lbl_img.set_label_colors(self.label_config)
            lbl_img.set_camera_id(cam_id)
            lbl_img.setStyleSheet("background-color: #111; border: 1px solid #444;")
            lbl_img.setMinimumSize(320, 240)
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setScaledContents(True)

            lbl_img.selection_finished.connect(
                lambda x, y, w, h, cid=cam_id: self._on_box_drawn(cid, x, y, w, h)
            )

            lbl_img.hovered.connect(
                lambda x, y, cid=cam_id: self._on_pixel_hovered(cid, x, y)
            )

            lbl_img.right_clicked.connect(self._on_label_right_clicked)

            v_layout.addWidget(lbl_title)
            v_layout.addWidget(lbl_img)

            self.strip_layout.addWidget(cam_box)
            self.image_labels[cam_id] = lbl_img

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _on_box_drawn(self, cam_id, x, y, w, h):
        """Helper to detect Shift key and emit signal."""
        modifiers = QApplication.keyboardModifiers()
        is_override = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

        # emit with the override flag
        self.box_drawn.emit(cam_id, x, y, w, h, is_override)

    def _on_label_right_clicked(self, cam_id: str):
        """Handler for when a specific camera label is double-clicked."""
        pixmap = self.image_labels[cam_id].pixmap()
        # Grab the projected boxes!
        projections = self.image_labels[cam_id].get_2d_projections()

        if pixmap:
            self.open_camera_modal(pixmap, cam_id, projections)

    def on_frame_update(self, data: FrameData) -> None:
        for cam_id, img_array in data.images.items():
            if cam_id in self.image_labels:
                h, w, c = img_array.shape
                self.image_labels[cam_id].set_original_resolution(w, h)
                self._update_single_camera(cam_id, img_array)

    def _update_single_camera(self, cam_id: str, array: np.ndarray):
        h, w, c = array.shape
        bytes_per_line = c * w
        q_img = QImage(array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_labels[cam_id].setPixmap(QPixmap.fromImage(q_img))

    def reset(self):
        for lbl in self.image_labels.values():
            lbl.clear()
            lbl.setText("No Data")

    def update_3d_projections(self, boxes: List[BoundingBox3D], calibration_dict: dict):
        """
        Passes 3D boxes and calibration to each camera label for live projection.
        """
        if not calibration_dict:
            return

        for cam_id, label_widget in self.image_labels.items():
            if cam_id in calibration_dict:
                calib = calibration_dict[cam_id]

                label_widget.set_projection_data(
                    boxes, calib.get("intrinsic"), calib.get("extrinsic")
                )

    def open_camera_modal(self, pixmap, cam_name, projections):
        """Triggered by double-clicking a camera image."""

        # Pass the projections into the modal
        modal = CameraPopOutModal(pixmap, cam_name, projections, parent=self)

        # Execute it (this blocks the main window until they hit Save or Cancel)
        if modal.exec() == QDialog.DialogCode.Accepted:
            # Retrieve the box they drew
            box_rect = modal.get_bounding_box()

            if box_rect:
                # Fetch the true physical resolution of the raw NumPy array
                orig_w = self.image_labels[cam_name].orig_width
                orig_h = self.image_labels[cam_name].orig_height

                # Calculate the ratio between the raw array and Qt's logical Pixmap
                scale_x = orig_w / pixmap.width()
                scale_y = orig_h / pixmap.height()

                # Apply scaling and safely clamp to the image boundaries
                # (This completely prevents SAM2/OpenCV array slicing crashes)
                x1 = max(0, int(box_rect.left() * scale_x))
                y1 = max(0, int(box_rect.top() * scale_y))
                x2 = min(orig_w, int(box_rect.right() * scale_x))
                y2 = min(orig_h, int(box_rect.bottom() * scale_y))

                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                # Prevent sending 0-pixel microscopic boxes to the AI
                if w < 5 or h < 5:
                    logger.warning(f"Box drawn on {cam_name} is too small to process.")
                    return

                logger.info(
                    f"Modal 2D Box -> Numpy Scaled [{cam_name}]: X:{x}, Y:{y}, W:{w}, H:{h}"
                )

                # Check for Shift modifier in case the backend relies on it
                modifiers = QApplication.keyboardModifiers()
                is_override = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

                # Fire the exact same signal the main window relies on!
                self.box_drawn.emit(cam_name, x, y, w, h, is_override)

    def _on_pixel_hovered(self, cam_id: str, x: int, y: int):
        pixmap = self.image_labels[cam_id].pixmap()
        if not pixmap or not hasattr(self.image_labels[cam_id], "orig_width"):
            return

        orig_w = self.image_labels[cam_id].orig_width
        orig_h = self.image_labels[cam_id].orig_height

        # Scale UI pixel to Raw Image Pixel
        widget_w = self.image_labels[cam_id].width()
        widget_h = self.image_labels[cam_id].height()

        if widget_w == 0 or widget_h == 0:
            return

        scale_x = orig_w / widget_w
        scale_y = orig_h / widget_h

        u = x * scale_x
        v = y * scale_y

        self.pixel_hovered.emit(cam_id, u, v)
