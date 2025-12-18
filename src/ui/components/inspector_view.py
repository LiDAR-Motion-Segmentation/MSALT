from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QGroupBox,
)
from PyQt6.QtCore import pyqtSignal
from src.core.objects import BoundingBox3D


class InspectorWidget(QWidget):
    box_changed = pyqtSignal(BoundingBox3D)

    def __init__(self) -> None:
        super().__init__()
        self.current_box = None
        self.is_updating = False
        self._init_ui()

    def _create_spinner(
        self, min_val=-100.0, max_val=100.0, step=0.05
    ) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(min_val, max_val)
        s.setSingleStep(step)
        s.setDecimals(3)
        s.valueChanged.connect(self._on_change)
        return s

    def _init_ui(self):
        layout = QVBoxLayout(self)
        grp_id = QGroupBox("Identity")
        form_id = QFormLayout(grp_id)

        self.spin_id = QSpinBox()
        self.spin_id.setRange(1, 9999)
        self.spin_id.valueChanged.connect(self._on_change)

        self.combo_cls = QComboBox()
        self.combo_cls.addItems(["moving_people", "static_people", "unkown"])
        self.combo_cls.currentTextChanged.connect(self._on_change)

        form_id.addRow("Track ID:", self.spin_id)
        form_id.addRow("Class:", self.combo_cls)
        layout.addWidget(grp_id)

        grp_geo = QGroupBox("Geometry (meters)")
        form_geo = QFormLayout(grp_geo)

        self.spin_x = self._create_spinner()
        self.spin_y = self._create_spinner()
        self.spin_z = self._create_spinner()

        self.spin_dx = self._create_spinner(0.1, 20.0, 0.05)
        self.spin_dy = self._create_spinner(0.1, 20.0, 0.05)
        self.spin_dz = self._create_spinner(0.1, 20.0, 0.05)

        self.spin_rot = self._create_spinner(-3.14159, 3.14159, 0.1)

        form_geo.addRow("Pos X:", self.spin_x)
        form_geo.addRow("Pos Y:", self.spin_y)
        form_geo.addRow("Pos Z:", self.spin_z)
        form_geo.addRow("Scale X:", self.spin_dx)
        form_geo.addRow("Scale Y:", self.spin_dy)
        form_geo.addRow("Scale Z:", self.spin_dz)
        form_geo.addRow("Heading (Rad):", self.spin_rot)

        layout.addWidget(grp_geo)
        layout.addStretch()

    def set_box(self, box: BoundingBox3D):
        if box is None:
            self.setEnabled(False)
            return

        self.current_box = box
        self.setEnabled(True)
        self.is_updating = True

        # Identity
        self.spin_id.setValue(box.track_id)
        idx = self.combo_cls.findText(box.label)
        if idx >= 0:
            self.combo_cls.setCurrentIndex(idx)

        # Geomtery
        self.spin_x.setValue(box.x)
        self.spin_y.setValue(box.y)
        self.spin_z.setValue(box.z)
        self.spin_dx.setValue(box.dx)
        self.spin_dy.setValue(box.dy)
        self.spin_dz.setValue(box.dz)
        self.spin_rot.setValue(box.heading)

        self.is_updating = False

    def _on_change(self):
        if self.is_updating or self.current_box is None:
            return

        self.current_box.track_id = self.spin_id.value()
        self.current_box.label = self.combo_cls.currentText()

        self.current_box.x = self.spin_x.value()
        self.current_box.y = self.spin_y.value()
        self.current_box.z = self.spin_z.value()

        self.current_box.dx = self.spin_dx.value()
        self.current_box.dy = self.spin_dy.value()
        self.current_box.dz = self.spin_dz.value()

        self.current_box.heading = self.spin_rot.value()

        self.box_changed.emit(self.current_box)

    def on_frame_update(self, data):
        self.set_box(None)
