# 3D LiDAR View

The 3D LiDAR View is the central, largest panel in the MSALT interface. Powered by `pyqtgraph.opengl`, this high-performance viewport is where you will spend the majority of your time inspecting point clouds and manipulating 3D bounding boxes.

## Navigation Controls

Moving around the 3D viewport is entirely mouse-driven:
* **Rotate (Orbit):** Left-Click and drag anywhere in the empty space to orbit the camera around the center focus point.
* **Pan:** Hold `Shift` + Left-Click and drag to slide the camera's focus point horizontally or vertically.
* **Zoom:** Use the Mouse Scroll Wheel to zoom in and out. 

## Visualizing the Point Cloud

To make it easier to distinguish between the road, vehicles, and the sky without relying solely on geometry, MSALT applies a dynamic **Z-Height Color Gradient** to the raw LiDAR points. 
* Points near the ground plane (e.g., `-2.0m`) are colored deep blue.
* Points near the sensor origin (`0.0m`) transition to cyan.
* Higher points (e.g., `+3.0m` like trees or traffic signs) transition to white.

Furthermore, when an AI model or a user draws a 3D bounding box, the points strictly contained *inside* that box are dynamically recolored based on the object's class label or selection state (e.g., bright yellow when selected).

## Bounding Boxes & Text Overlays

MSALT renders Oriented Bounding Boxes (OBBs) as 3D wireframes. 
* **Track IDs & Labels:** To keep track of objects, MSALT continuously calculates the projection of each 3D box's center point onto your 2D computer monitor. It then paints a crisp, highly readable text overlay (e.g., `18: moving_car`) directly above the box, ensuring the text is always facing the camera and perfectly scaled regardless of how you rotate the 3D view.

## Interacting with the View

The viewport is not just a passive display; it is a fully interactive canvas.

### Selecting Objects
To select a box, simply **Left-Click** on it. 
* Under the hood, MSALT shoots a mathematical ray from your mouse cursor through the 3D space and uses the `Slab Method` to determine which bounding box you clicked.
* Clicking a box highlights it in yellow and instantly populates its exact dimensions in the right-hand Inspector panel.

### Drawing Objects
To manually create a new bounding box, hold **`Ctrl` + Left-Click**. 
* This triggers MSALT's custom 3-click drawing state machine. 
* Your first click establishes the origin on the ground plane, the second defines the length and heading, and the third defines the width. You can then drag or scroll to extrude the box's height. 
* *(For a complete guide on this process, see the [Manual 3D Drawing](../workflows/manual_drawing.md) section).*

### Ground Truth Comparison
If your loaded dataset contains pre-annotated ground truth data (like the NuScenes dataset), a **"Compare Ground Truth"** checkbox will appear at the top of the LiDAR view.
* Toggling this on will overlay the dataset's official bounding boxes in bright **Magenta**.
* Points inside these ground truth boxes will also flash magenta, allowing you to visually verify the accuracy of your AI's auto-annotations against the dataset's official labels.

- also in the `perf/nuscenes` branch you can see the ground truth boxes too

![alt text](../assets/gt_comparison.png)