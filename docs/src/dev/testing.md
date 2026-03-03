# Testing

- We use `pytest` for logic verification and `ruff` for linting
```bash
# Check code style
uv run ruff check . --fix

# Run the test suite
uv run pytest 
```

### annotation_manager tests (`test_annotation_manager.py`)
- `test_add_box_assigns_track_id_when_unset`: verifies auto track_id assignment when track_id = -1.
- `test_add_box_preserves_existing_track_id`: ensures explicit IDs are not overwritten.
- `test_delete_box_removes_box_from_frame`: simple delete behavior per frame.
- `test_remove_box_by_track_id`: checks remove_box deletes the correct track ID and leaves others.
- `test_deselect_all_clears_selected_flag_across_frames`: ensures deselect_all clears selected across all frames.

### commands tests (`test_commands.py`)
- Uses a lightweight FakeAnnotationManager to avoid disk and UI coupling.
- `test_add_box_command_execute_and_undo`: AddBoxCommand correctly adds and undoes.
- `test_delete_box_command_execute_and_undo`: DeleteBoxCommand deletes and restores.
- `test_bulk_delete_command_execute_and_undo`: BulkDeleteCommand deletes a batch and undo restores all.
- `test_modify_box_command_execute_undo_and_redo`: validates ModifyBoxCommand now:
1. replaces old_state with new_state on execute,
2. restores old_state on undo,
3. reapplies new_state on redo.
- To support this, `ModifyBoxCommand` in `commands.py` was fixed so `execute()` uses new_state and `undo()` restores old_state.

### geometry tests (`test_geometry.py`)
- Existing tests kept as is (`test_box_corners`, `test_points_in_box`).
- New tests:
1. `test_interpolate_box_midpoint`: checks that interpolate_box at `t=0.5` produces the geometric midpoint and halfway heading.
2. `test_refine_heading_returns_current_when_too_few_points`: ensures refine_heading returns the original heading for very small point sets (<5).

### UI components tests (`test_ui_components.py`)
- Uses an offscreen Qt platform (QT_QPA_PLATFORM="offscreen") and dummy classes (DummyViewWidget, DummyScatter) to test CameraStripWidget and LidarVisualizer without the overhead or instability of full GUI rendering.

1. `test_camera_strip_frame_update_sets_resolution_and_pixmap`: Verifies that when a new frame is loaded, the camera strip correctly updates the original image dimensions and sets the Qt pixmap.

2. `test_camera_strip_box_signal_includes_shift_override`: Mocks the Qt keyboard modifiers to simulate holding the Shift key, ensuring the box_drawn signal correctly emits a True flag for the shift override.

3. `test_camera_strip_update_3d_projection_updates_only_calibrated_camera`: Ensures that 3D projection updates strictly apply intrinsic and extrinsic matrices to cameras that have calibration data, correctly ignoring uncalibrated views.

4. `test_lidar_on_frame_update_sets_ground_plane_and_draws_when_no_boxes`: Validates that loading a new point cloud calculates the ground plane Z-height (using the 50th percentile minus a bias) and triggers the default point rendering when no boxes exist.

5. `test_lidar_draw_points_default_sets_scatter_data`: Checks that the default rendering correctly passes the point cloud coordinates, sizes, and base colors to the pyqtgraph scatter plot item.

6. `test_lidar_update_boxes_recolors_points_and_rebuilds_line_items`: Mocks the point-in-box math to verify that points inside a bounding box are dynamically recolored to match the label's color map. It also ensures old 3D box lines are cleared and new ones are added to the view widget.
