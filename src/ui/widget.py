#!/usr/bin/env python3
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import Qimage, QPixmap
import numpy as np
from typing import Optional

class CameraWidget(QWidget):
    """
    A reusable widget to display a single 2D camera feed.
    
    Attributes:
        camera_id (str): Identifier for the camera (e.g., 'CAM_FRONT').
    """
    
    def __init__(self, camera_id: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        self.layout = QVBoxLayout(self)
        self.layout.setContentMargins(2,2,2,2)
        self.label_title = QLabel(self.camera_id)