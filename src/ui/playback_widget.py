from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QStyle
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


class PlaybackWidget(QWidget):
    """
    A reusable timeline control widget.

    Signals:
        frame_changed (int): Emitted when the current frame index changes.
    """

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._current_frame = 0
        self._is_playing = False

        # timer for auto-playback
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_timer_tick)
        self._fps = 10

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        # Play/Pause Button (Uses standard icons)
        self.btn_play = QPushButton()
        self.btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_play.setToolTip("Play/Pause (Space)")
        self.btn_play.clicked.connect(self.toggle_playback)
        layout.addWidget(self.btn_play)

        # Frame Counter Label (e.g., "0 / 100")
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setMinimumWidth(80)
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_frame.setStyleSheet("font-family: monospace; font-weight: bold;")
        layout.addWidget(self.lbl_frame)

        # The Timeline Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.sliderMoved.connect(self.set_current_frame)  # User drags
        self.slider.valueChanged.connect(self.set_current_frame)  # Programmatic change
        layout.addWidget(self.slider)

    def setup_timeline(self, total_frames: int):
        # Initializes the slider range based on loaded data
        self._total_frames = total_frames
        self.slider.setMaximum(total_frames - 1)
        self.slider.setValue(0)
        self._update_label()

    def toggle_playback(self):
        # switch between play and pause states
        self._is_playing = not self._is_playing
        if self._is_playing:
            self.btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )
            self._play_timer.start(int(1000 / self._fps))
        else:
            self.btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )
            self._play_timer.stop()

    def _on_timer_tick(self):
        # Advance frame automatically
        next_frame = (self._current_frame + 1) % self._total_frames
        self.set_current_frame(next_frame)

    def set_current_frame(self, index: int):
        if index < 0 or index >= self._total_frames:
            return

        if index == self._current_frame:
            return

        self._current_frame = index

        # Block signals on slider to prevent infinite loop if slider triggered this
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

        self._update_label()

        # EMIT SIGNAL -> The Main Window listens to this
        self.frame_changed.emit(index)

    def _update_label(self):
        self.lbl_frame.setText(f"{self._current_frame} / {self._total_frames}")
