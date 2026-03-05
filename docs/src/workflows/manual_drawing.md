# Manual 3D Bounding Box Drawing

While MSALT's auto-annotation pipeline handles the bulk of the work, severely occluded objects or sparse point clouds may require manual intervention.

MSALT features a highly optimized, 3-click state machine that allows you to draw and extrude Oriented Bounding Boxes (OBBs) directly inside the 3D LiDAR space.

## The 3-Click Workflow
Ensure the 3D LiDAR view is active and you have selected your desired label (e.g., static_car) from the right-hand Inspector panel.

### Step 1: Define the Base (Footprint)
The base of the bounding box is defined using `Ctrl + Left Click` to plot points on the ground plane.

1. **Click 1 (Origin)**: Hold `Ctrl` and `Left Click` on the ground near the center of the object. A yellow dot will appear.

2. **Click 2 (Length & Heading)**: Hold `Ctrl and Left Click` again to define the length and the orientation (yaw/heading) of the object.

- Pro-Tip: The axis is "magnetic." If you drag near 0°, 90°, 180°, or 270°, the line will automatically snap to a perfect grid alignment!

3. **Click 3 (Width)**: Hold `Ctrl` and `Left Click` a third time to define the width. A cyan ghost box will immediately appear on the screen.

### Step 2: Extrude the Height
Once the base is confirmed (after the 3rd click), the system automatically transitions into height-adjustment mode.

- **Scroll Wheel**: Scroll up or down to grow or shrink the height of the box.

- **Click & Drag**: Alternatively, you can click and drag the mouse vertically to smoothly scale the box height.

### Step 3: Finalize
To lock the box in place and save it to the `AnnotationManager`:

- Hold Ctrl + Left Click one final time.

The box will change to its assigned class color (e.g., Blue for static_car) and will immediately project its 2D coordinates onto the camera strip above.

### Canceling an Action
If you make a mistake while plotting points or adjusting the height, simply press the `Esc` key. This will instantly kill the drawing state, erase the cyan ghost box, and return you to standard camera navigation without saving any data.

### Modifying Existing Boxes
If you need to tweak a manually drawn box later:

1. Click the box in the 3D view (or select it from the right-hand Annotation List).

2. Use the Inspector Panel to manually adjust the precise X, Y, Z coordinates, Scale, or Heading (in radians).

3. Alternatively, press R to run the PCA Refinement algorithm, which will attempt to automatically snap the heading to the nearest cluster of points.