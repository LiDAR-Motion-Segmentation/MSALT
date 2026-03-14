from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from src.core.objects import BoundingBox3D
from src.data.structures import FrameData
from typing import List, Dict


class AnnotationListWidget(QWidget):
    # Signals
    box_selected = pyqtSignal(BoundingBox3D)  # When user clicks a row
    box_deleted = pyqtSignal(BoundingBox3D)  # When user clicks delete
    label_changed = pyqtSignal(BoundingBox3D, str)  # When combo changes

    def __init__(self, label_config: List[Dict]):
        super().__init__()
        self.current_boxes: list = []
        self.label_config = label_config or [
            {"name": "Unknown", "color": [0, 255, 0], "hotkey": "0"}
        ]
        self.color_map = {item["name"]: item["color"] for item in self.label_config}
        self._init_ui()
        self._setup_hotkeys()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)

        # Label Selector
        lbl_layout = QHBoxLayout()
        lbl_layout.addWidget(QLabel("Active Label :"))
        self.combo_label = QComboBox()
        for item in self.label_config:
            self.combo_label.addItem(item["name"])
        lbl_layout.addWidget(self.combo_label)
        self.layout.addLayout(lbl_layout)

        # list of objects
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.layout.addWidget(self.list_widget)

        # Delete Button
        btn_del = QPushButton("Delete Selected (Del)")
        btn_del.setStyleSheet(
            """
            QPushButton {
                background-color: #c0392b; 
                color: white; 
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:pressed { background-color: #a93226; }
        """
        )
        btn_del.clicked.connect(self.delete_selected)
        self.layout.addWidget(btn_del)

    def _setup_hotkeys(self):
        for i, item in enumerate(self.label_config):
            key = item.get("hotkey")
            if key:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(
                    lambda idx=i: self.set_active_label_index(idx)
                )

    def set_active_label_index(self, index):
        if 0 <= index < self.combo_label.count():
            self.combo_label.setCurrentIndex(index)
            self.label_changed.emit(self.combo_label.currentText())

    def get_color_rgb(self, label_name: str) -> List[int]:
        """Returns [R, G, B] list. Defaults to Green."""
        return self.color_map.get(label_name, [0, 255, 0])

    def update_list(self, boxes: list[BoundingBox3D]):
        self.list_widget.clear()
        self.current_boxes = boxes

        for box in boxes:
            label_text = f"ID {box.track_id}: {box.label}"
            item = QListWidgetItem(label_text)
            # Store the actual object reference in the item for easy access
            item.setData(Qt.ItemDataRole.UserRole, box)
            self.list_widget.addItem(item)

    def _on_item_clicked(self, item):
        box = item.data(Qt.ItemDataRole.UserRole)
        self.box_selected.emit(box)

    def delete_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.item(row)
            box = item.data(Qt.ItemDataRole.UserRole)
            self.box_deleted.emit(box)

            # Optimistically remove from view
            self.list_widget.takeItem(row)

    def get_current_label(self) -> str:
        return self.combo_label.currentText()

    def on_frame_update(self, data: FrameData) -> None:
        pass
