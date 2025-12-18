from tkinter import Widget
from typing import List
from PyQt6.QtWidgets import QMainWindow, QDockWidget, QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeyEvent, QKeySequence, QShortcut
from pathlib import Path

from src.data.data_controller import DataController
from src.ui.interfaces import BasePluginWidget
from src.ui.components.camera_view import CameraStripWidget
from src.ui.components.lidar_view import LidarVisualizer
from src.ui.playback_widget import PlaybackWidget
from src.ui.components.annotation_list import AnnotationListWidget
from src.ui.components.inspector_view import InspectorWidget

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
        base_out = Path(self.data_controller.cfg.output.dir)
        self.annotation_manager.load_frames(
            boxes_dir=base_out / "3d",
            meta_dir=base_out / "metadata"
        )
        
        self.seg_engine = SegmentationEngine(self.data_controller.cfg.models)

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
        
        # Camera Strip
        cam_ids = self.data_controller.get_camera_ids()
        self.cam_widget = CameraStripWidget(cam_ids)
        self.add_dock(self.cam_widget, "Cameras", Qt.DockWidgetArea.TopDockWidgetArea)

        # LiDAR View (Central focused)
        self.lidar_widget = LidarVisualizer()
        # We set LiDAR as the Main Central Widget for maximum space
        self.setCentralWidget(self.lidar_widget)

        # Playback Controls (Bottom Dock)
        self.playback = PlaybackWidget()
        self.playback.setup_timeline(self.data_controller.get_total_frames())

        # A Dock at the bottom is best for timeline.
        dock_timeline = QDockWidget("Timeline", self)
        dock_timeline.setWidget(self.playback)
        dock_timeline.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_timeline)
        
        # Shortcut for saving
        save_action = QAction("Save Annotations", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_current_work)
        self.addAction(save_action)
        
        # Annotation List Dock
        self.list_panel = AnnotationListWidget()
        self.add_dock(self.list_panel, "Annotations", Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Connect Signals
        self.list_panel.box_selected.connect(self.on_box_selected)
        self.list_panel.box_deleted.connect(self.on_box_deleted)
        
        # Right Arrow -> Next Frame
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_next.activated.connect(self.next_frame)

        # Left Arrow -> Previous Frame
        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_prev.activated.connect(self.prev_frame)
        
        # Spacebar -> Play/Pause
        self.shortcut_play = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_play.activated.connect(self.toggle_play)
        
        # Inspector dock
        self.inspector = InspectorWidget()
        self.add_dock(self.inspector, "Inspector", Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Connect: When Inspector changes a value, refresh the 3D view
        self.inspector.box_changed.connect(self.on_box_edited)
        
    def save_current_work(self):
        """Saves both 3D JSON and Metadata JSON using 000000.json format."""
    
        base_out = Path(self.data_controller.cfg.output.dir)
        boxes_dir = base_out / "3d"
        meta_dir = base_out / "metadata"
        
        if hasattr(self.data_controller, 'pcd_files'):
            try:
                pcd_path = self.data_controller.pcd_files[self.current_frame_idx]
                current_frame_name = pcd_path.stem
            except IndexError:
                pass
            
        filename = f"{self.current_frame_idx:06d}.json"
        
        self.annotation_manager.save_frame(
            self.current_frame_idx, 
            boxes_dir,
            meta_dir, 
            filename
        )
        
        self.statusBar().showMessage(f"Saved: {filename}", 3000)
        logger.info(f"Exported annotation to: {filename}")

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
        self.current_frame_idx = idx
        self.current_frame_data = self.data_controller.get_frame(idx)
        boxes = self.annotation_manager.get_boxes(idx)
        boxes_2d_map = defaultdict(list)
        
        # prepare 2D Box Map for Camera View
        for box in boxes:
            if box.source_2d:
                cam_id = box.source_2d['cam_id']
                rect = box.source_2d['rect']
                boxes_2d_map[cam_id].append({
                    'rect': rect,
                    'id': box.track_id,
                    'label': box.label
                })
                
        # update for plugins
        for plugin in self.plugins:
            plugin.on_frame_update(self.current_frame_data)
        
        self.cam_widget.update_2d_boxes(boxes_2d_map)    
        self.lidar_widget.on_frame_update(self.current_frame_data)
        self.lidar_widget.update_boxes(boxes)
        self.list_panel.update_list(boxes)
        
    def on_box_selected(self, box):
        # Highlight the box in 3D view when clicked in list.
        # Deselect all
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        for b in current_boxes:
            b.selected = (b == box)
        # Redraw
        self.lidar_widget.update_boxes(current_boxes)
        
        # update inspector panel
        self.inspector.set_box(box)
        
    def on_box_edited(self, box):
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        self.lidar_widget.update_boxes(current_boxes)
        self.list_panel.update_list(current_boxes)
        self.save_current_work()
        
    def on_box_deleted(self, box):
        # Remove box from manager and refresh.
        self.annotation_manager.delete_box(self.current_frame_idx, box)
        self.load_frame(self.current_frame_idx) # Refresh view
        
        # Auto-save after delete 
        self.save_current_work()

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
            new_box.label = self.list_panel.get_current_label()

            # Save indices for Red Coloring
            new_box.point_indices = np.where(mask_3d)[0]
            
            # Save 2D Rect for Cyan Box
            new_box.source_2d = {'cam_id': cam_id, 'rect': [x, y, w, h]}
            
            # Save and Refresh
            self.annotation_manager.add_box(self.current_frame_idx, new_box)
            self.load_frame(self.current_frame_idx)  # Redraw UI
            self.save_current_work()
            
            # self.debug_draw_frustum(cam_id, [x, y, w, h])
            logger.info(
                f"Created Box at {new_box.x:.2f}, {new_box.y:.2f}, {new_box.z:.2f}"
            )
        else:
            logger.warning("No 3D points found inside the mask.")
            
    def keyPressEvent(self, event: QKeyEvent) -> None:
        # right arrow -> Next Frame
        if event.key() == Qt.Key.Key_Right:
            total_frames = self.data_controller.get_total_frames()
            if self.current_frame_idx < total_frames - 1:
                self.load_frame(self.current_frame_idx + 1)
                # Update slider position visually
                self.playback.slider.setValue(self.current_frame_idx)
                
        # left arrow -> Previous Frame
        elif event.key() == Qt.Key.Key_Left:
            if self.current_frame_idx > 0:
                self.load_frame(self.current_frame_idx - 1)
                self.playback.slider.setValue(self.current_frame_idx)
        
        else:
            super().keyPressEvent(event)
            
    def next_frame(self):
        total = self.data_controller.get_total_frames()
        if self.current_frame_idx < total - 1:
            new_idx = self.current_frame_idx + 1
            self.load_frame(new_idx)
            # Sync the slider
            self.playback.slider.setValue(new_idx)

    def prev_frame(self):
        if self.current_frame_idx > 0:
            new_idx = self.current_frame_idx - 1
            self.load_frame(new_idx)
            # Sync the slider
            self.playback.slider.setValue(new_idx)
            
    def toggle_play(self):
        # Assuming PlaybackWidget has a toggle method, or you simulate the button click
        if hasattr(self.playback, 'play_btn'):
            self.playback.play_btn.click()
