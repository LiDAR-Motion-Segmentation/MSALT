from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal, Qt


class AutomationPanel(QWidget):
    propagate_requested = pyqtSignal()
    interpolate_requested = pyqtSignal()
    grid_view_requested = pyqtSignal()

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
        self.btn_interp = QPushButton("Interpolate (I)")
        self.btn_interp.setToolTip("Linearly fill gaps between previous frame and this one")
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

        layout.addStretch()
        self.setLayout(layout)
        
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
