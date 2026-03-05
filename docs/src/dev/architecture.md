# System Architecture

- MSALT follows a modular Model-View-Controller (MVC) pattern to separate UI logic from geometric processing.

![alt text](../assets/flowchart_v3.png)

MSALT is built on a modular, event-driven architecture using **Python** and **PyQt6** for the user interface, **PyQtGraph / OpenGL** for 3D rendering, and **PyTorch / Ultralytics** for the AI inference engine. 

To keep the codebase maintainable and memory-efficient, the system strictly separates the presentation layer (UI), data management, and core mathematical/AI logic.

## High-Level Directory Structure

The project is organized into three primary domains within the `src/` directory:

1. **`src/ui/` (The View):** Contains all PyQt6 widgets, windows, and rendering logic.
2. **`src/data/` (The Model):** Handles dataset loading, caching, and serving raw frames.
3. **`src/core/` (The Controller/Engine):** Houses the AI models, 3D geometry math, state management, and command history.


## Core Modules

### 1. The UI Layer (`src/ui`)
The UI is orchestrated by `MainWindow`. It acts as the central hub, initializing the plugins and routing signals between the timeline, the AI engine, and the renderers.
* **`LidarVisualizer` (`lidar_view.py`):** Uses `pyqtgraph.opengl` to render the 3D point cloud. It maintains a custom state machine to handle the 3-click manual box drawing process and converts 2D screen clicks into 3D raycasts.
* **`CameraStripWidget` & `CameraPopOutModal`:** Display the 2D camera feeds. They handle user bounding-box interactions and perform HiDPI coordinate mapping to ensure drawn boxes map correctly to the raw NumPy image arrays.
* **`AutomationPanel`:** Houses the trigger buttons for YOLO+SAM2 Auto-Annotation, Kalman Tracking, and SAM2 Interpolation.

### 2. The Data Layer (`src/data`)
* **`DataController`:** Acts as the single source of truth for the active dataset. It manages the `RealsenseLoader` (or custom NuScenes loaders) to fetch images, point clouds, and calibration matrices (intrinsics and extrinsics) on demand.

### 3. Core Engine & AI (`src/core`)
This is the mathematical brain of MSALT.
* **`AnnotationManager`:** Maintains the state of all 3D bounding boxes in the current session. It handles the serialization/deserialization of annotations to JSON files and assigns unique track IDs.
* **`SegmentationEngine`:** A wrapper around the Ultralytics library that safely manages GPU memory. It loads `YOLOv8` for 2D object detection and `SAM2` for precise pixel masking.
* **`GeometryUtils`:** A completely stateless class containing static methods for heavy 3D math. It handles:
    * 2D-to-3D Frustum filtering and Raycasting.
    * 3D-to-2D image projection.
    * `fit_box_to_cloud` (DBSCAN + OBB fitting) and PCA heading refinement.
* **`CommandHistory`:** Implements the classic "Command Design Pattern." Every action (e.g., `AddBoxCommand`, `BulkDeleteCommand`) is encapsulated here to enable robust Undo `(Ctrl+Z)` and `Redo (Ctrl+Y)` functionality.

## Application Data Flow

To understand how MSALT operates, here is the lifecycle of an **Auto-Annotation** event:

1. **Trigger:** The user clicks "Auto-Annotate" in the `AutomationPanel`.
2. **Detection:** `MainWindow` asks the `SegmentationEngine` to run YOLO on the active `FrameData` images, returning 2D bounding boxes.
3. **Masking:** Each 2D box is passed to SAM2 to extract a precise boolean pixel mask.
4. **Projection:** `GeometryUtils` projects the entire 3D LiDAR cloud onto the 2D image plane using the camera's calibration matrices. Points landing inside the SAM2 mask are extracted.
5. **Fitting:** The extracted 3D points are passed to `GeometryUtils.fit_box_to_cloud`, which uses DBSCAN to remove noise and fits an Oriented Bounding Box (OBB).
6. **State Update:** The new `BoundingBox3D` object is validated (checked for duplicates) and pushed to the `AnnotationManager` via an `AddBoxCommand`.
7. **Render:** `MainWindow` calls `refresh_views_only()`, forcing the `LidarVisualizer` and `CameraStripWidget` to draw the new box.

## Configuration System

MSALT uses **Hydra** for configuration management. Instead of hardcoding parameters, everything is defined in the `config/` directory.

The `MainWindow` ingests these configurations at startup:
* **`config.yaml`:** Defines UI window sizes, global hotkeys, output directories, and DBSCAN math parameters (epsilon, min_samples).
* **`models/default.yaml`:** Dictates which AI weights to load (e.g., `yolov8n.pt`, `sam2_b.pt`), inference confidence thresholds, and specific COCO class targets.