from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox, QLabel
from PyQt6.QtCore import pyqtSignal, Qt


class AutomationPanel(QWidget):
    propagate_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title / Header
        title = QLabel("Automation Tools")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        layout.addWidget(title)

        # Group: Sequential Tracking
        grp_seq = QGroupBox("Sequential Tracking")
        vbox_seq = QVBoxLayout(grp_seq)

        # Propagate Button
        self.btn_propagate = QPushButton("Propagate Selection")
        self.btn_propagate.setToolTip(
            "Copy selected box to the next frame and auto-fit points."
        )

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
        self.btn_propagate.clicked.connect(self.propagate_requested.emit)

        vbox_seq.addWidget(self.btn_propagate)
        layout.addWidget(grp_seq)
        layout.addStretch()

    def on_frame_update(self, data):
        """
        Required by MainWindow plugin system.
        The automation panel doesn't need to update visuals per frame, 
        so we leave this empty.
        """
        pass
