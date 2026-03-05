# Data Controller

In MSALT's Model-View-Controller (MVC) architecture, the `DataController` serves as the central Model. It is responsible for managing the massive influx of multi-sensor data, ensuring that images, 3D point clouds, and calibration matrices are efficiently loaded, synchronized, and served to the UI and AI engines.

## Core Responsibilities

The `DataController` acts as the single source of truth for the active dataset. Its primary responsibilities include:

1. **Frame Management:** It determines the total number of frames in a sequence and handles sequential or random-access loading when the user scrubs the timeline.
2. **Sensor Synchronization:** It ensures that when you request a specific frame, you receive the exact LiDAR sweep and the corresponding RGB camera images that were captured at that exact microsecond.
3. **Calibration Serving:** It loads and serves the intrinsic (`K`) and extrinsic (pose) matrices for each camera, which are absolutely critical for the 2D-to-3D backprojection math.

## The `FrameData` Object

When the `MainWindow` requests a specific frame index (e.g., `data_controller.get(idx)`), the controller does not return raw loose files. Instead, it constructs and returns a standardized `FrameData` object.

This object acts as a structured payload containing:
* **`point_cloud`:** A NumPy array containing the raw LiDAR points.
* **`images`:** A dictionary mapping camera IDs to their respective RGB image arrays.
* **`metadata`:** A dictionary containing frame-specific information, most importantly the `calibration` block containing the `intrinsic` and `extrinsic` matrices for every camera. 

## Memory Management & Caching

Because an uncompressed LiDAR point cloud and 5-6 high-resolution RGB images can consume hundreds of megabytes per frame, loading them all into RAM at once would instantly crash standard workstations. 

To solve this, the `DataController` employs **Lazy Loading and Caching**:
* **Lazy Loading:** Data is only read from the disk when the user actually navigates to that frame.
* **LRU Caching:** The controller keeps the most recently viewed frames in an LRU (Least Recently Used) cache. If you scrub back one frame, it loads instantly from RAM. If you scrub back 50 frames, the controller intelligently flushes the old memory and loads the new files from disk.

## Building Custom Loaders

Out of the box, MSALT supports datasets like NuScenes and standard RealSense bags. However, one of the most powerful features of the `DataController` is its extensible **Loader Interface**. 

If you build a custom robot or drone with a unique sensor configuration, you do not need to rewrite the MSALT UI or AI logic. You simply create a new Loader class that inherits from the base interface. 

Your custom loader only needs to implement two things:
1. Parse your custom folder structure or `.bag` file.
2. Package the data into the standard `FrameData` format expected by the `DataController`.

By dropping your custom loader into `src/data/loaders/` and updating your Hydra configuration, MSALT will instantly be able to visualize, annotate, and auto-track data from your custom hardware.