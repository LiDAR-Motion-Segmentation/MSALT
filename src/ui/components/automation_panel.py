from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QHBoxLayout,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal, Qt


class AutomationPanel(QWidget):
    propagate_requested = pyqtSignal()
    interpolate_requested = pyqtSignal()
    grid_view_requested = pyqtSignal()
    tracking_requested = pyqtSignal()
    yolo_requested = pyqtSignal()
    point_size_changed = pyqtSignal(int)
    open_analytics_requested = pyqtSignal()
    export_segmentation_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title / Header
        title = QLabel("<b>Automation Tools</b>")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title)

        # Propagate Button
        self.btn_propagate = QPushButton("Propagate Selection (P) ")
        self.btn_propagate.setToolTip(
            "Copy selected box to the next frame and auto-fit points."
        )
        self.btn_propagate.clicked.connect(self.propagate_requested.emit)
        # Style it to look like an "Action" button (Blue/Green)
        self.btn_propagate.setStyleSheet(
            """
            QPushButton {
                background-color: #2980b9; 
                color: white; 
                padding: 8px; 
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:pressed { background-color: #1abc9c; }
        """
        )
        layout.addWidget(self.btn_propagate)

        # interpolate button
        self.btn_interp = QPushButton("SAM2 Interpolation (I)")
        self.btn_interp.setToolTip(
            "Linearly fill gaps between previous frame and this one"
        )
        self.btn_interp.clicked.connect(self.interpolate_requested.emit)

        self.btn_interp.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #E0E0E0;
            }
            QPushButton:pressed {
                background-color: #B0B0B0;
            }
        """)
        layout.addWidget(self.btn_interp)

        # Kalman filter
        self.btn_linear = QPushButton("Kalman Filter Tracking (K)")
        self.btn_linear.setStyleSheet("""
            QPushButton {
                background-color: #008080;  /* Teal */
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #20B2AA; } /* Light Sea Green */
        """)
        self.btn_linear.clicked.connect(self.tracking_requested.emit)
        layout.addWidget(self.btn_linear)

        self.btn_yolo = QPushButton("Auto-Annotate (YOLO+SAM2)")
        self.btn_yolo.setToolTip("Run YOLO detection and project 3D boxes")
        self.btn_yolo.clicked.connect(self.yolo_requested.emit)
        self.btn_yolo.setStyleSheet("""
            QPushButton {
                background-color: #e67e22; /* Orange */
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #d35400; }
        """)
        layout.addWidget(self.btn_yolo)

        self.btn_export_seg = QPushButton("Export 3D Segmentation (.label)")
        self.btn_export_seg.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0; /* A nice purple for Deep Tech features */
                color: white; 
                font-weight: bold; 
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #BA68C8; }
        """)
        self.btn_export_seg.clicked.connect(self._on_export_seg_clicked)
        layout.addWidget(self.btn_export_seg)

        layout.addStretch()
        self.setLayout(layout)

        size_layout = QHBoxLayout()
        lbl_size = QLabel("PCD Point Size:")
        lbl_size.setStyleSheet("color: #AAA;")

        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 10)  # Limit between 1 and 10 pixels
        self.spin_point_size.setValue(2)  # Default size
        self.spin_point_size.valueChanged.connect(self.point_size_changed.emit)

        size_layout.addWidget(lbl_size)
        size_layout.addWidget(self.spin_point_size)

        layout.addLayout(size_layout)

        self.btn_analytics = QPushButton("QA Analytics and Telemetry")
        self.btn_analytics.setStyleSheet("""
            QPushButton {
                background-color: #8BC34A; 
                color: #111111; 
                font-weight: bold; 
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #9CCC65;
            }
        """)
        self.btn_analytics.clicked.connect(self.open_analytics_requested.emit)

        layout.addWidget(self.btn_analytics)

        self.btn_grid = QPushButton("Batch View (B)")
        self.btn_grid.clicked.connect(self.grid_view_requested.emit)
        self.btn_grid.setStyleSheet("""
            QPushButton {
                background-color: #FFD700; /* Gold/Yellow */
                color: black;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #FFC107; /* Darker Yellow on hover */
            }
            QPushButton:pressed {
                background-color: #FFB300;
            }
        """)
        layout.addWidget(self.btn_grid)

    def on_frame_update(self, data):
        """
        Required by MainWindow plugin system.
        The automation panel doesn't need to update visuals per frame,
        so we leave this empty.
        """
        pass

    def _on_export_seg_clicked(self):
        """Opens a dialog to pick a folder, then emits the path."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory for .label files"
        )
        if dir_path:  # If the user didn't hit cancel
            self.export_segmentation_requested.emit(dir_path)
