# UI Overview &amp; Layout

MSALT is designed with a modular, dock based interface. It maximizes the central workspace for 3D point cloud manipulation while keeping all synchronized sensor feeds, AI automation tools, and geometric controls immediately accessible on the periphery.

![alt text](../assets/ui_v13.png)

## The Main Panels

The interface is divided into five primary regions:

### 1. Top Panel: The Camera Strip
The top dock displays the synchronized 2D RGB camera feeds for the current frame. 
* **Live Projection:** 3D bounding boxes drawn in the LiDAR view are instantly mathematically projected onto these camera views.
* **Interaction:** You can left click and drag to draw manual 2D bounding boxes, or **Right-Click** any camera to open the high resolution **Pop-Out Modal** for precise pixel-level annotation.

### 2. Center Workspace: 3D LiDAR Visualizer
The heart of MSALT. This is a high-performance OpenGL viewport that renders the raw 3D LiDAR point cloud.
* **Navigation:** Click and drag to rotate the camera. Scroll to zoom. Hold `Shift` while dragging to pan.
* **Drawing:** Hold `Ctrl` and Left-Click to manually plot and extrude 3D Oriented Bounding Boxes (OBBs) using the 3-click workflow.
* **Ground Truth Comparison:** A checkbox at the top left of this viewport allows you to toggle the rendering of magenta Ground Truth boxes (if loaded in the dataset metadata) to visually verify your AI predictions.

### 3. Left Panel: Automation Tools
This panel houses your 1 click AI pipelines.
* **Propagate Selection (P):** Copies the selected box to the next frame and snaps it to the new LiDAR points.
* **SAM2 Interpolate (I):** Uses SAM2 pixel masking to automatically track and fit a box across a sequence of frames.
* **Kalman Filter Tracking (K):** Uses a velocity based linear Kalman Filter to predict an object's future trajectory.
* **Auto-Annotate (YOLO+SAM2):** The primary macro. Runs YOLO detection across all cameras, generates SAM2 masks, and instantly back-projects them into the 3D point cloud.
* **Batch View (B):** Opens a grid window to inspect a specific track ID across multiple frames simultaneously.
* **QA Analytics and Telemetry:** Opens a window with graphs for completeness, class distribution, consistency and validity 

### 4. Right Panel: Annotations & Inspector
This dual-purpose dock gives you granular control over your data.
* **Annotation List:** Displays the currently active label (selected via a dropdown) and a list of all Track IDs currently present in the frame. Clicking an ID here highlights it in the 3D view.
* **Inspector:** When a box is selected, this panel displays its exact mathematical properties. You can manually type in adjustments to its Position (`X, Y, Z`), Scale (`dx, dy, dz`), or Heading (Yaw in radians).

### 5. Bottom Panel: Timeline & Playback
The timeline allows you to scrub through the dataset sequence.
* Contains a slider spanning the total frames loaded by the Data Controller.
* Use the **Left** and **Right Arrow Keys** to quickly step forward and backward frame by frame without using the mouse.