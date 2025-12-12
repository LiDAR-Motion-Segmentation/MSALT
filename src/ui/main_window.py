from tkinter import Widget
from typing import List
from PyQt6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

from src.data.data_controller import DataController
from src.ui.interfaces import BasePluginWidget
from src.ui.components.camera_view import CameraStripWidget
from src.ui.components.lidar_view import LidarVisualizer
from src.ui.playback_widget import PlaybackWidget

class MainWindow(QMainWindow):
    def __init__(self, data_controller: DataController):
        super().__init__()
        self.setWindowTitle("SALT: Sensor Fusion Annotator")
        self.resize(1920, 1080)
        self.data_controller = data_controller
        
        # registery of active plugins
        self.plugins: List[BasePluginWidget] = []
        
        self._init_ui()
        self._connect_signals()
        
        if self.data_controller.get_total_frames() > 0:
            self.load_frame(0)
            
    def _init_ui(self):
        # assembling UI using dock widgets
        
        # 1. Central Widget (Maybe a summary or empty for now, Docks are the main actors)
        # Usually, the LiDAR view is the 'Central' widget
        self.central_panel = QWidget()
        self.setCentralWidget(self.central_panel)
        
        # We can hide the central widget if we want everything docked
        # self.central_panel.hide()
        
        # 2. Initialize Plugins
        # A. Camera Strip
        cam_ids = self.data_controller.get_camera_ids()
        self.cam_widget = CameraStripWidget(cam_ids)
        self.add_dock(self.cam_widget, "Cameras", Qt.DockWidgetArea.TopDockWidgetArea)
        
        # B. LiDAR View (Central focused)
        self.lidar_widget = LidarVisualizer()
        # We set LiDAR as the Main Central Widget for maximum space
        self.setCentralWidget(self.lidar_widget) 
        # Note: Since we set it as central, we don't add it to self.plugins list 
        # automatically if we rely on that list for updates.
        self.plugins.append(self.lidar_widget)
        self.plugins.append(self.cam_widget)
        
        # 3. Playback Controls (Bottom Dock)
        self.playback = PlaybackWidget()
        self.playback.setup_timeline(self.data_controller.get_total_frames())
        
        # A Dock at the bottom is best for timeline.
        dock_timeline = QDockWidget("Timeline", self)
        dock_timeline.setWidget(self.playback)
        dock_timeline.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_timeline)
        
    def add_dock(self, widget: BasePluginWidget, title: str, area: Qt.DockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self.plugins.append(widget)
        
    def _connect_signals(self):
        # Wiring the Playback -> Controller -> UI.
        self.playback.frame_changed.connect(self.load_frame)
        
    def load_frame(self, idx: int):
        """
        Orchestrator:
        1. Fetch Data
        2. Notify ALL Plugins
        """
        frame_data = self.data_controller.get_frame(idx)
        for plugin in self.plugins:
            plugin.on_frame_update(frame_data)