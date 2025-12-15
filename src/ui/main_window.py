from tkinter import Widget
from typing import List
from PyQt6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt

from src.data.data_controller import DataController
from src.ui.interfaces import BasePluginWidget
from src.ui.components.camera_view import CameraStripWidget
from src.ui.components.lidar_view import LidarVisualizer
from src.ui.playback_widget import PlaybackWidget

from src.core.annotation_manager import AnnotationManager
from src.core.objects import BoundingBox3D
from src.core.geometry import GeometryUtils
from src.core.segmentation import SegmentationEngine
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, data_controller: DataController):
        super().__init__()
        self.setWindowTitle("SALT: Sensor Fusion Annotator")
        self.resize(1920, 1080)
        self.data_controller = data_controller
        self.annotation_manager = AnnotationManager()
        self.seg_engine = SegmentationEngine(self.data_controller.cfg.models)

        # trying dummy data
        # dummy_box = BoundingBox3D(x=0, y=0, z=-1, dx=4, dy=2, dz=1.5, heading=0.5)
        # self.annotation_manager.add_box(0, dummy_box)

        # State tracking
        self.current_frame_idx = 0
        self.current_frame_data = None  # Cache the data for math ops

        # registery of active plugins
        self.plugins: List[BasePluginWidget] = []

        self._init_ui()
        self._connect_signals()

        if self.data_controller.get_total_frames() > 0:
            self.load_frame(0)

    def _init_ui(self):
        # assembling UI using dock widgets

        # # 1. Central Widget (Maybe a summary or empty for now, Docks are the main actors)
        # # Usually, the LiDAR view is the 'Central' widget
        # self.central_panel = QWidget()
        # self.setCentralWidget(self.central_panel)

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
        # self.plugins.append(self.lidar_widget)
        # self.plugins.append(self.cam_widget)

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
        self.cam_widget.box_drawn.connect(self.handle_annotation)

    def load_frame(self, idx: int):
        """
        Orchestrator:
        1. Fetch Data
        2. Notify ALL Plugins
        """
        self.current_frame_idx = idx
        self.current_frame_data = self.data_controller.get_frame(idx)
        boxes = self.annotation_manager.get_boxes(idx)
        boxes_2d_map = defaultdict(list)
        
        # prepare 2D Box Map for Camera View
        for box in boxes:
            if box.source_2d:
                cam_id = box.source_2d['cam_id']
                rect = box.source_2d['rect']
                boxes_2d_map[cam_id].append(rect)
                
        # update for plugins
        for plugin in self.plugins:
            plugin.on_frame_update(self.current_frame_data)
        
        self.cam_widget.update_2d_boxes(boxes_2d_map)    
        self.lidar_widget.on_frame_update(self.current_frame_data)
        self.lidar_widget.update_boxes(boxes)

    def handle_annotation(self, cam_id: str, x: int, y: int, w: int, h: int):
        # Logic: Box -> SAM2 Mask -> 3D Projection -> Box Fit
        if (
            self.current_frame_data is None
            or self.current_frame_data.point_cloud is None
        ):
            logger.warning("Cannot annotate: No point cloud data available.")
            return

        calib = self.current_frame_data.metadata.get("calibration", {}).get(cam_id)
        if not calib:
            logger.error(f"No calibration found for {cam_id}")
            return

        K = calib["intrinsic"]
        
        # This is Cam->World pose
        camera_pos = calib["extrinsic"]

        if K is None or camera_pos is None:
            logger.error(f"Calibration incomplete for {cam_id}")
            return

        # the actual RGB image array for the model
        image = self.current_frame_data.images.get(cam_id)
        if image is None:
            logger.error(f"Image not found for {cam_id}")
            return

        logger.info(f"Running SAM2 on {cam_id}...")
        
        # Generate Mask (AI Step)
        mask = self.seg_engine.get_mask_from_box(image, [x, y, w, h])

        # Filter Points (Frustum Culling)
        points = self.current_frame_data.point_cloud
        mask_3d = GeometryUtils.get_points_in_mask(points, mask, K, camera_pos)

        selected_points = points[mask_3d]
        logger.info(
            f"Annotation: Selected {len(selected_points)} points inside 2D box."
        )

        # fit 3D box
        box_params = GeometryUtils.fit_box_to_cloud(selected_points)

        if box_params:
            # Create the Box Object
            new_box = BoundingBox3D(**box_params)
            new_box.label = "Person"

            # Save indices for Red Coloring
            new_box.point_indices = np.where(mask_3d)[0]
            
            # Save 2D Rect for Cyan Box
            new_box.source_2d = {'cam_id': cam_id, 'rect': [x, y, w, h]}
            
            # Save and Refresh
            self.annotation_manager.add_box(self.current_frame_idx, new_box)
            self.load_frame(self.current_frame_idx)  # Redraw UI
            # self.debug_draw_frustum(cam_id, [x, y, w, h])
            logger.info(
                f"Created Box at {new_box.x:.2f}, {new_box.y:.2f}, {new_box.z:.2f}"
            )
        else:
            logger.warning("No 3D points found inside the mask.")
