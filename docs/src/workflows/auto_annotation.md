# Auto-Annotation (YOLOv8 + SAM2)

The Auto-Annotation pipeline is the core automation feature of MSALT. It bridges the gap between 2D camera images and 3D LiDAR point clouds, allowing you to automatically generate accurate 3D bounding boxes for pedestrians, cars, and other vehicles across all camera views with a single click.

## How to Use It
- `Load a Frame`: Ensure your current frame has both LiDAR and Camera data loaded.
- `Trigger Pipeline`: Click the orange Auto-Annotate (YOLO+SAM2) button in the left-hand Automation panel.
- `Review` : The system will freeze for a moment while the neural networks process the images. Once complete, the 3D LiDAR view will automatically populate with bounding boxes, and the status bar will report how many unique objects were created.

## Under the Hood: The 2D-to-3D Pipeline
To keep VRAM usage low and execution fast, MSALT uses a highly optimized, 6-step forward-projection pipeline.

1. `2D Object Detection (YOLO)`
The pipeline begins by passing every available camera image into a YOLO object detection model. YOLO scans the 2D image and outputs rough bounding boxes [x, y, w, h] for configured classes (e.g., cars, pedestrians, trucks).

2. `Precise Pixel Masking (SAM2)`
Because a 2D bounding box includes a lot of background pixels (like the road or sky), we cannot use it directly to grab 3D points. MSALT passes the YOLO bounding box and the image into the Segment Anything Model 2 (SAM2). SAM2 returns a strict, pixel-perfect boolean mask of the exact object.

3. `Forward-Projection & Frustum Culling`
-Instead of shooting complex 3D rays from the camera into the point cloud, MSALT uses a faster mathematical approach:
-It takes the entire 3D LiDAR point cloud and projects it onto the 2D image plane using the camera's intrinsic (K) and extrinsic pose matrices.
-Any LiDAR point that lands perfectly inside the 2D SAM mask is "kept".

4. `Ego-Vehicle Rejection`
Because the LiDAR is mounted on the roof of the ego-vehicle, it often captures reflections from the vehicle's own hood or trunk. To prevent the AI from spawning bounding boxes on the ego-vehicle, MSALT calculates the physical distance of the new box from the sensor origin (0, 0). Any box falling within a 3.0-meter radius of the LiDAR sensor is immediately rejected.

5. `3D Box Fitting (DBSCAN + PCA)`
- The filtered 3D points are passed to the Geometry engine
- DBSCAN is used to cluster the points and remove outliers (like stray leaves or rain).
- PCA (Principal Component Analysis) is used to determine the primary orientation (heading/yaw) of the object.
- If the point cloud is too sparse (fewer than 5 points), the system falls back to a mathematical raycast, placing a standard-sized box at the correct depth on the ground plane.

6. `3D Spatial Deduplication (Bird's-Eye NMS)`
- Because multiple cameras have overlapping fields of view, the same physical object might be detected twice.
- Before saving a new 3D box, MSALT performs a spatial check against all existing boxes in the frame. It calculates the 2D Euclidean distance in the Bird's-Eye View (X, Y plane). If the new box is within 0.8 meters of an existing box, it is classified as a cross-camera duplicate and discarded.

## Configuration
You can configure the behavior of the Auto-Annotation pipeline in your `models/default.yaml` and `config.yaml` files:

- `Model Selection`: Swap between yolov8n.pt (Nano, best for 6GB VRAM) or yolov8l.pt (Large, best accuracy).
- `Detection Thresholds`: Adjust the conf_threshold to make YOLO more or less strict.
- `Allowed Classes`: Modify the COCO index array (e.g., [0, 2, 3, 5, 7]) to control which objects the system annotates automatically.