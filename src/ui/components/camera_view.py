from typing import Container, List, Dict
import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QVBoxLayout, QScrollArea
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

from src.ui.interfaces import BasePluginWidget
from src.data.structures import FrameData

class CameraStripWidget(BasePluginWidget):
    def __init__(self, camera_ids: List[str]):
        super().__init__(title="camera strip")
        self.camera_ids = camera_ids
        self.image_labels: Dict[str, QLabel] = {}
        self._setup_ui()
        
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
            lbl_img = QLabel()
            lbl_img.setStyleSheet("background-color: #111; border: 1px solid #444;")
            lbl_img.setMinimumSize(320, 240)
            lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_img.setText("No Signal")
            
            v_layout.addWidget(lbl_title)
            v_layout.addWidget(lbl_img)
            
            self.strip_layout.addWidget(cam_box)
            self.image_labels[cam_id] = lbl_img
            
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
    def on_frame_update(self, data: FrameData) -> None:
        for cam_id, img_array in data.images.items():
            if cam_id in self.image_labels:
                self._update_single_camera(cam_id, img_array)
                
    def _update_single_camera(self, cam_id: str, array: np.ndarray):
        h, w, c = array.shape
        bytes_per_line = c * w
        q_img = QImage(array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        target_label = self.image_labels[cam_id]
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            target_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        target_label.setPixmap(scaled_pixmap)
        
    def reset(self):
        for lbl in self.image_labels.values():
            lbl.clear()
            lbl.setText("No Data")