from typing import List
from PyQt6.QtWidgets import QMainWindow, QDockWidget
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
from src.ui.components.automation_panel import AutomationPanel
from src.ui.components.batch_view import BatchGridWindow

from src.core.annotation_manager import AnnotationManager
from src.core.objects import BoundingBox3D
from src.core.geometry import GeometryUtils
from src.core.segmentation import SegmentationEngine
from src.core.commands import (
    CommandHistory, 
    AddBoxCommand, 
    BulkDeleteCommand
)

import logging
import numpy as np
import time
import traceback

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, data_controller: DataController):
        super().__init__()
        self.setWindowTitle("MSALT: Multi Sensor Fusion Annotator")
        self.resize(1920, 1080)
        self.data_controller = data_controller
        
        # Hydra configuration shortcuts
        self.cfg = self.data_controller.cfg
        self.geometry_cfg = getattr(self.cfg, "geometry", None)
        self.automation_cfg = getattr(self.cfg, "automation", None)
        self.lidar_view_cfg = getattr(self.cfg, "lidar_view", None)
        
        # Initialize Manager and Load Frames
        self.annotation_manager = AnnotationManager()
        base_out = Path(self.cfg.output.dir)
        self.annotation_manager.load_frames(
            boxes_dir=base_out / "3d", meta_dir=base_out / "metadata"
        )

        self.seg_engine = SegmentationEngine(self.cfg.models)
        self.labels_cfg= getattr(self.cfg, "labels", None)

        # Initializing the history
        self.history = CommandHistory()

        # State tracking
        self.current_frame_idx = 0
        self.current_frame_data = None  # Cache the data for math ops

        # registery of active plugins
        self.plugins: List[BasePluginWidget] = []

        self._init_ui()
        self._connect_signals()

        if self.data_controller.get_total_frames() > 0:
            self.load_frame(0)
            
        self.session_start_time = time.time()
        self.objects_created_session = 0


    def _init_ui(self):
        
        # CENTER: LiDAR View 
        self.lidar_widget = LidarVisualizer(cfg=self.lidar_view_cfg)
        self.lidar_widget.set_label_colors(self.labels_cfg)
        self.setCentralWidget(self.lidar_widget)
        self.lidar_widget.view_widget.box_created.connect(self.handle_3d_annotation)

        # TOP: Camera Strip 
        cam_ids = self.data_controller.get_camera_ids()
        self.cam_widget = CameraStripWidget(cam_ids)
        self.cam_widget.set_label_config(self.labels_cfg)
        self.add_dock(self.cam_widget, "Cameras", Qt.DockWidgetArea.TopDockWidgetArea)

        # LEFT: Automation Panel 
        self.automation_panel = AutomationPanel()
        self.add_dock(
            self.automation_panel, "Automation", Qt.DockWidgetArea.LeftDockWidgetArea
        )
        # Connect Signals
        self.automation_panel.propagate_requested.connect(self.propagate_selection)
        self.automation_panel.interpolate_requested.connect(self.interpolate_selection)

        # BOTTOM: Playback 
        self.playback = PlaybackWidget()
        self.playback.setup_timeline(self.data_controller.get_total_frames())
        dock_timeline = QDockWidget("Timeline", self)
        dock_timeline.setWidget(self.playback)
        dock_timeline.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock_timeline)

        # RIGHT: Annotation List 
        self.list_panel = AnnotationListWidget(label_config=self.labels_cfg)
        self.add_dock(
            self.list_panel, "Annotations", Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.list_panel.box_selected.connect(self.on_box_selected)
        self.list_panel.box_deleted.connect(self.on_box_deleted)

        # RIGHT: Inspector 
        self.inspector = InspectorWidget(label_config=self.labels_cfg)
        self.add_dock(
            self.inspector, "Inspector", Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.inspector.box_changed.connect(self.on_box_edited)
        
        # Initialize the Grid Window (Hidden by default)
        self.batch_grid_window = BatchGridWindow(self.data_controller, self.annotation_manager)
        self.batch_grid_window.request_jump.connect(self.load_frame)
        self.batch_grid_window.data_modified.connect(self.save_specific_frame)
        self.automation_panel.grid_view_requested.connect(self.open_grid_view)
        
        # Save (Ctrl+S)
        save_action = QAction("Save Annotations", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_current_work)
        self.addAction(save_action)
        
        # Undo (Ctrl+Z) 
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo_action)
        self.addAction(undo_action)
        
        # Redo (Ctrl+Y)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y")) 
        redo_action.triggered.connect(self.redo_action)
        self.addAction(redo_action)

        # Navigation (Arrows)
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.shortcut_next.activated.connect(self.next_frame)

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.shortcut_prev.activated.connect(self.prev_frame)

        # Playback (Space)
        self.shortcut_play = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut_play.activated.connect(self.toggle_play)

        # Delete (Del) - Global Context
        self.shortcut_del = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        self.shortcut_del.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.shortcut_del.activated.connect(self.delete_selection)
        
        # Automation Shortcuts (P, I)
        self.shortcut_p = QShortcut(QKeySequence("P"), self)
        self.shortcut_p.activated.connect(self.propagate_selection)
        self.shortcut_interp = QShortcut(QKeySequence("I"), self)
        self.shortcut_interp.activated.connect(self.interpolate_selection)
        
        # Refine heading (R)
        self.shortcut_refine = QShortcut(QKeySequence("R"), self)
        self.shortcut_refine.activated.connect(self.refine_selection)
        
        # Batch view (B)
        self.shortcut_batch = QShortcut(QKeySequence("B"), self)
        self.shortcut_batch.activated.connect(self.open_grid_view)
        
    def save_current_work(self):
        """Saves both 3D JSON and Metadata JSON using 000000.json format."""

        base_out = Path(self.data_controller.cfg.output.dir)
        boxes_dir = base_out / "3d"
        meta_dir = base_out / "metadata"

        filename = f"{self.current_frame_idx:06d}.json"

        self.annotation_manager.save_frame(
            self.current_frame_idx, boxes_dir, meta_dir, filename
        )

        self.statusBar().showMessage(f"Saved: {filename}", 3000)
        logger.info(f"Exported annotation to: {filename}")
        
        elapsed = time.time() - self.session_start_time
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        count = len(current_boxes)
        
        if count > 0:
            avg_time = elapsed / count
            self.statusBar().showMessage(f"Saved. Efficiency: {avg_time:.2f} sec/object", 5000)
            logger.info(f"BENCHMARK: Frame {self.current_frame_idx}, Objects: {count}, Time: {elapsed:.2f}s, Avg: {avg_time:.2f}s/obj")
        
    def save_specific_frame(self, frame_idx: int):
        """
        Saves a specific frame index to disk immediately.
        Triggered by Batch View auto-save.
        """
        base_out = Path(self.data_controller.cfg.output.dir)
        boxes_dir = base_out / "3d"
        meta_dir = base_out / "metadata"

        # Construct the correct filename for the specific frame
        filename = f"{frame_idx:06d}.json"

        self.annotation_manager.save_frame(
            frame_idx, boxes_dir, meta_dir, filename
        )
        logger.debug(f"Auto-saved Frame {frame_idx}")

    def add_dock(self, widget: BasePluginWidget, title: str, area: Qt.DockWidgetArea):
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        self.plugins.append(widget)

    def _connect_signals(self):
        """ Wiring the Playback -> Controller -> UI. """
        self.playback.frame_changed.connect(self.load_frame)
        self.cam_widget.box_drawn.connect(self.handle_annotation)
        
    def open_grid_view(self):
        """Opens the Grid Window for the currently selected track."""
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        selected = [b for b in current_boxes if b.selected]
        
        if not selected:
            self.statusBar().showMessage("Select a box to view in Grid.", 2000)
            return
            
        track_id = selected[0].track_id
        
        # Show Window
        self.batch_grid_window.show()
        self.batch_grid_window.raise_() # Bring to front
        
        # Load Data (Center on current frame, show 16 frames context)
        self.batch_grid_window.load_track(track_id, self.current_frame_idx, window_size=18)

    def load_frame(self, idx: int):
        self.current_frame_idx = idx
        self.current_frame_data = self.data_controller.get(idx)
        boxes = self.annotation_manager.get_boxes(idx)
        
        # Explicitly deselect all boxes when entering a new frame.
        for b in boxes:
            b.selected = False
        

        # update for plugins
        for plugin in self.plugins:
            plugin.on_frame_update(self.current_frame_data)
            
        self.lidar_widget.on_frame_update(self.current_frame_data)
        self.lidar_widget.update_boxes(boxes)
        
        # Pass the FULL list of 3D boxes and the calibration dict
        if self.current_frame_data.metadata and "calibration" in self.current_frame_data.metadata:
            calib = self.current_frame_data.metadata["calibration"]
            self.cam_widget.update_3d_projections(boxes, calib)
            
        self.list_panel.update_list(boxes)
        
        # Update Grid if open
        if self.batch_grid_window.isVisible():
            pass
        
        # Reset timer for time per frame metric
        self.session_start_time = time.time()

    def on_box_selected(self, box):
        # Highlight the box in 3D view when clicked in list.
        # Deselect all
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        for b in current_boxes:
            b.selected = b == box
        # Redraw
        self.lidar_widget.update_boxes(current_boxes)

        # update inspector panel
        self.inspector.set_box(box)

    def on_box_edited(self, box):
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        self.lidar_widget.update_boxes(current_boxes)
        self.list_panel.update_list(current_boxes)
        self.save_current_work()
        self.refresh_views_only()

    def on_box_deleted(self, box):
        # Remove box from manager and refresh.
        self.annotation_manager.delete_box(self.current_frame_idx, box)
        self.load_frame(self.current_frame_idx)  # Refresh view

        # Auto-save after delete
        self.save_current_work()
        
    def on_camera_box_drawn(self, cam_id: str, x: int, y: int, w: int, h: int, is_override: bool):
        """ Update the slot definition to accept the bool"""
        self.handle_annotation(cam_id, x, y, w, h, is_override)

    def handle_annotation(self, cam_id: str, x: int, y: int, w: int, h: int, is_override: bool):
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        selected_box = next((b for b in current_boxes if b.selected), None)
        
        if selected_box and is_override:
            selected_box.visual_overrides[cam_id] = [x, y, w, h]
            
            logger.info(f"Updated visual override for Track {selected_box.track_id} on {cam_id}")
            self.statusBar().showMessage(f"Visuals aligned for Track {selected_box.track_id}", 2000)
            
            # Refresh to show the new box
            self.refresh_views_only()
            self.save_current_work()
            return
        
        if selected_box and not is_override:
            # We are about to create a new box, but one is selected.
            self.annotation_manager.deselect_all()
            
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
        
        try: 
            T_camera_lidar = camera_pos
        except np.linalg.LinAlgError:
            logger.error("Singular matrix in calibration")
            return
        
        R_camera_lidar = T_camera_lidar[:3, :3]
        forward_vector = R_camera_lidar @ np.array([0, 0, 1])
        camera_heading = np.arctan2(forward_vector[1], forward_vector[0])
        
        # Pinhole Ray: (u-cx)/fx, (v-cy)/fy, 1.0
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        
        center_x, center_y = x + w/2, y + h/2
        ray_cam = np.array([(center_x - cx)/fx, (center_y - cy)/fy, 1.0])
        
        # Transform Ray to Lidar Frame
        ray_lidar = R_camera_lidar @ ray_cam
        cam_origin = T_camera_lidar[:3, 3]
        
        fallback_center = None
        
        if abs(ray_lidar[2]) > 0.05: # Avoid divide by zero (horizon)
            t = (-1.0 - cam_origin[2]) / ray_lidar[2]
            if t > 0:
                fallback_center = cam_origin + t * ray_lidar

        # the actual RGB image array for the model
        image = self.current_frame_data.images.get(cam_id)
        if image is None:
            logger.error(f"Image not found for {cam_id}")
            return

        logger.info(f"Running SAM2 on {cam_id}...")

        # Generate Mask (AI Step)
        try:
            mask = self.seg_engine.get_mask_from_box(image, [x, y, w, h])
        except Exception as e:
            logger.error(f"SAM 2 Error: {e}")
            mask = None
            
        if mask is None:
            self.statusBar().showMessage("SAM 2 could not find an object there.", 2000)
            return

        # Filter Points (Frustum Culling)
        points = self.current_frame_data.point_cloud
        mask_3d = GeometryUtils.get_points_in_mask(points, mask, K, camera_pos)

        selected_points = points[mask_3d]
        logger.info(
            f"Annotation: Selected {len(selected_points)} points inside 2D box."
        )

        # fit 3D box
        try:
            current_label = self.list_panel.get_current_label()
            box_params = GeometryUtils.fit_box_to_cloud(selected_points,
                                                        label=current_label, 
                                                        camera_heading=camera_heading,
                                                        fallback_center=fallback_center
                                                        )
            
            if box_params is None:
                logger.warning("Fit failed: Points too sparse and fallback Raycast failed.")
                self.statusBar().showMessage("Could not fit box. Try drawing closer.", 3000)
                return

            # Create the Box Object
            new_box = BoundingBox3D(**box_params)
            new_box.label = self.list_panel.get_current_label()

            # Save indices for Red Coloring
            new_box.point_indices = np.where(mask_3d)[0]

            # Save 2D Rect for Cyan Box
            new_box.source_2d = {"cam_id": cam_id, "rect": [x, y, w, h]}

            # Save and Refresh
            cmd = AddBoxCommand(
                self.annotation_manager, 
                self.current_frame_idx, 
                new_box
            )
            self.history.push(cmd)
            
            self.load_frame(self.current_frame_idx)  # Redraw UI
            self.save_current_work()

            # self.debug_draw_frustum(cam_id, [x, y, w, h])
            logger.info(
                f"Created Box at {new_box.x:.2f}, {new_box.y:.2f}, {new_box.z:.2f}"
            )
        except Exception as e:
            logger.error(f"box parameter is failing : {e}")
            traceback.print_exc()

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
        if hasattr(self.playback, "play_btn"):
            self.playback.play_btn.click()

    def propagate_selection(self):
        """Entry point: User pressed P or clicked button."""
        # Find what is selected
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        selected_boxes = [b for b in current_boxes if b.selected]

        if not selected_boxes:
            self.statusBar().showMessage("Please select a box to propagate.", 2000)
            return

        # Run the copy logic
        self._perform_propagation(selected_boxes)

    def _perform_propagation(self, boxes_to_copy):
        """Copies the list of boxes to the next frame."""
        next_idx = self.current_frame_idx + 1
        total_frames = self.data_controller.get_total_frames()

        if next_idx >= total_frames:
            self.statusBar().showMessage("Already at the last frame!", 2000)
            return

        # Get Data for Next Frame (To calculate new points/2D box)
        next_data = self.data_controller.get(next_idx)

        count = 0
        for old_box in boxes_to_copy:

            # Clone the Box (Deep Copy)
            new_box = BoundingBox3D(
                track_id=old_box.track_id,  # Keep ID same
                label=old_box.label,        # Keep Label same
                x=old_box.x,
                y=old_box.y,
                z=old_box.z,
                dx=old_box.dx,
                dy=old_box.dy,
                dz=old_box.dz,
                heading=old_box.heading,
            )

            # Recalculate 3D Points (Red/Blue Coloring)
            if next_data.point_cloud is not None:
                indices = GeometryUtils.get_points_in_box(
                    next_data.point_cloud, new_box
                )
                new_box.point_indices = indices

            # Recalculate 2D Box (Cyan Visual)
            # We try to use the same camera ID as the previous frame
            if old_box.source_2d and next_data.images:
                cam_id = old_box.source_2d.get("cam_id")

                # Check if this camera exists in the next frame
                if (
                    cam_id in next_data.images
                    and cam_id in next_data.metadata["calibration"]
                ):
                    calib = next_data.metadata["calibration"][cam_id]
                    img_shape = next_data.images[cam_id].shape

                    # Math Logic
                    res = GeometryUtils.project_box_to_image(
                        new_box, calib["extrinsic"], calib["intrinsic"], img_shape
                    )

                    if res:
                        new_box.source_2d = {"cam_id": cam_id, "rect": res}

            # Save to Manager
            self.annotation_manager.add_box(next_idx, new_box)
            count += 1

        # Jump to the next frame so user can see the result
        self.load_frame(next_idx)
        self.save_current_work()
        self.statusBar().showMessage(
            f"Propagated {count} objects to Frame {next_idx}", 2000
        )

    def delete_selection(self):
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        to_delete = [b for b in current_boxes if b.selected]
        
        if not to_delete:
            self.statusBar().showMessage("No box selected to delete.", 2000)
            return
        
        # Perform Delete
        cmd = BulkDeleteCommand(
            self.annotation_manager, 
            self.current_frame_idx, 
            to_delete
        )
        self.history.push(cmd)
            
        # Refresh View
        self.load_frame(self.current_frame_idx)
        self.save_current_work()
        self.statusBar().showMessage(f"Deleted {len(to_delete)} objects.", 2000)
        
    def interpolate_selection(self):
        """
        Action for the 'Interpolate' button.
        """
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        selected_boxes = [b for b in current_boxes if b.selected]
        
        if not selected_boxes:
            self.statusBar().showMessage("Select a box to interpolate/track.", 2000)
            return
        
        total_filed = 0
        TRACK_HORIZON = 10
        start_frame = self.current_frame_idx
        end_frame = min(start_frame + TRACK_HORIZON, self.data_controller.get_total_frames() - 1)
        for box in selected_boxes:
            self.statusBar().showMessage(f"Tracking ID {box.track_id}...")
            count = self.annotation_manager.run_smart_interpolation(
                box.track_id, 
                self.current_frame_idx, 
                end_frame,
                self.data_controller,
                self.seg_engine
            )
            total_filed += count
            
        if total_filed > 0:
            self.statusBar().showMessage(f"Saving {total_filed} frames to disk...", 3000)
            
            # Retrieve output paths from Config, matching save_current_work()
            base_out = Path(self.data_controller.cfg.output.dir)
            boxes_dir = base_out / "3d"
            meta_dir = base_out / "metadata"
            
            # Save Start Frame (Current)
            self.save_current_work()
            
            # Save Future Frames (Tracked)
            for f_idx in range(start_frame + 1, end_frame + 1):
                filename = f"{f_idx:06d}.json"
                
                # Force save from AnnotationManager
                self.annotation_manager.save_frame(
                    f_idx, 
                    boxes_dir, 
                    meta_dir, 
                    filename
                )
                logger.info(f"Auto-saved tracked frame: {filename}")
                
            self.load_frame(self.current_frame_idx)
            self.statusBar().showMessage(f"Tracking Complete. Jumped to Frame {end_frame}.", 4000)
        else:
            self.statusBar().showMessage("Tracking failed. No new boxes created.", 3000)
            
    def undo_action(self):
        msg = self.history.undo()
        if msg:
            self.load_frame(self.current_frame_idx)
            self.save_current_work()
            self.statusBar().showMessage(f"Undid: {msg}", 2000)
        else:
            self.statusBar().showMessage("Nothing to undo.", 1000)
            
    def redo_action(self):
        msg = self.history.redo()
        if msg:
            self.load_frame(self.current_frame_idx)
            self.save_current_work()
            self.statusBar().showMessage(f"Redid: {msg}", 2000)
    
    def refine_selection(self):
        """
        Applies PCA to the selected box to fix its rotation.
        """
        current_boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        selected_boxes = [b for b in current_boxes if b.selected]
        
        if not selected_boxes:
            self.statusBar().showMessage("Select a box to refine (R).", 2000)
            return
        
        if self.current_frame_data is None or self.current_frame_data.point_cloud is None:
            return
        
        count = 0
        points = self.current_frame_data.point_cloud
        
        for box in selected_boxes:
            indices = GeometryUtils.get_points_in_box(points, box)
            
            if len(indices) < 5:
                logger.warning(f"ID {box.track_id}: Not enough points to refine.")
                continue
            
            box_points = points[indices]
            new_heading = GeometryUtils.refine_heading(box_points, box.heading)
            old_heading = box.heading
            box.heading = float(new_heading)
            
            logger.info(f"Refined ID {box.track_id}: {old_heading:.2f} -> {new_heading:.2f}")
            count += 1
            
        if count > 0:
            self.save_current_work()
            
            # Force UI Refresh
            self.lidar_widget.update_boxes(current_boxes) 
            
            # Update Inspector values if a single box is selected
            if len(selected_boxes) == 1:
                self.inspector.set_box(selected_boxes[0])
            
            self.statusBar().showMessage(f"Refined {count} boxes via PCA.", 2000)
            
    def handle_3d_annotation(self, cx, cy, cz, dx, dy, dz, heading):
        # Ask Manager for a fresh ID
        track_id = self.annotation_manager.get_unique_id()
        
        # create box
        new_box = BoundingBox3D(
            track_id=track_id,
            x=cx,
            y=cy,
            z=cz,
            dx=dx,
            dy=dy,
            dz=dz,
            heading=heading,
            label=self.list_panel.get_current_label()
        )
        
        # add to manager
        self.annotation_manager.add_box(self.current_frame_idx, new_box)
        
        # select it
        new_box.selected = True
        
        # refresh
        self.load_frame(self.current_frame_idx)
        self.save_current_work()
        
        # switch to inspector for fine-tuning
        self.inspector.set_box(new_box)
        
    def refresh_views_only(self):
        """Refreshes overlays without reloading heavy PCD data."""
        boxes = self.annotation_manager.get_boxes(self.current_frame_idx)
        
        # update LiDAR
        self.lidar_widget.update_boxes(boxes)
        
        # update Cameras
        if self.current_frame_data:
            calib = self.current_frame_data.metadata.get("calibration", {})
            self.cam_widget.update_3d_projections(boxes, calib)
            
        # update List
        self.list_panel.update_list(boxes)