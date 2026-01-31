from typing import List, Dict
import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QScrollArea, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen

from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData
from src.ui.components.drawable_label import DrawableLabel
from src.core.objects import BoundingBox3D
        
class CameraStripWidget(BasePluginWidget):

    # New Signal: (CameraID, x, y, w, h)
    box_drawn = pyqtSignal(str, int, int, int, int)

    def __init__(self, camera_ids: List[str]):
        super().__init__(title="camera strip")
        self.camera_ids = camera_ids
        self.image_labels: Dict[str, DrawableLabel] = {}
        self._setup_ui()
        
    def set_label_config(self, label_config: List[Dict]):
        self.label_config = label_config
        
        # self.image_labels is a dict: {'CAM_FRONT': DrawableLabel, ...}
        if hasattr(self, 'image_labels'):
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
            lbl_title.setStyleSheet("font-weight: bold; color: #ccc;")

            # Image Placeholder
            lbl_img = DrawableLabel()
            if hasattr(self, 'label_config'):
                lbl_img.set_label_colors(self.label_config)
            lbl_img.set_camera_id(cam_id)
            lbl_img.setStyleSheet("background-color: #111; border: 1px solid #444;")
            lbl_img.setMinimumSize(320, 180)
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setScaledContents(False)
            lbl_img.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

            lbl_img.selection_finished.connect(
                lambda x, y, w, h, cid=cam_id: self.box_drawn.emit(cid, x, y, w, h)
            )

            v_layout.addWidget(lbl_title)
            v_layout.addWidget(lbl_img)

            self.strip_layout.addWidget(cam_box)
            self.image_labels[cam_id] = lbl_img

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

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
                    boxes,
                    calib.get('intrinsic'),
                    calib.get('extrinsic')
                )         
