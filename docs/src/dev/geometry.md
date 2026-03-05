# Geometry &amp; Raycasting Math

At the core of MSALT's sensor fusion capabilities is the `GeometryUtils` class. This entirely stateless, static class handles all matrix transformations, frustum culling, and point cloud analytics required to bridge the 2D camera world and the 3D LiDAR world.

This page breaks down the primary mathematical operations used throughout the application.

## 1. Coordinate Systems

MSALT must constantly translate between three distinct coordinate spaces:
1. **LiDAR / World Space (3D):** Origin `(0,0,0)` is the center of the LiDAR sensor on the ego-vehicle. Coordinates are in meters `(X, Y, Z)`.
2. **Camera Space (3D):** Origin is the camera lens. Z represents depth (forward), X is right, and Y is down.
3. **Image / Screen Space (2D):** Origin `(0,0)` is the top-left corner of the image or UI widget. Coordinates are in pixels `(u, v)`.



## 2. 3D-to-2D Projection (LiDAR to Image)

To project 3D LiDAR points onto a 2D camera image (used for visualizing frustums or filtering SAM2 masks), MSALT applies the standard Pinhole Camera Model.

1. **Extrinsic Transformation:** LiDAR points `(X, Y, Z)` are converted to homogeneous coordinates `(X, Y, Z, 1)` and multiplied by the inverted Extrinsic Matrix (Camera-to-World pose) to map them into the Camera's local coordinate frame.
2. **Depth Filtering:** Any point with a `Z <= 0` in the camera frame is behind the lens and is aggressively discarded (Frustum Culling).
3. **Intrinsic Projection:** The valid 3D points are multiplied by the Camera's Intrinsic Matrix `(K)`. The resulting `X` and `Y` values are divided by their `Z` depth to normalize them into flat 2D pixel coordinates `(u, v)`.

## 3. 2D-to-3D Raycasting (Screen to World)

When a user clicks on the 3D LiDAR view to manually draw a box, MSALT must convert that 2D screen click into a 3D physical location.

1. **NDC Conversion:** The `(X, Y)` screen pixel coordinates are mapped to OpenGL Normalized Device Coordinates (NDC) ranging from `-1.0` to `1.0`.
2. **Unprojection:** The NDC coordinates are extended to a near-plane `(z=-1)` and far-plane `(z=1)` point. These points are multiplied by the inverse of the Camera's View-Projection Matrix (`inv_mvp`) to determine their true 3D world positions.
3. **Ray Formation:** Subtracting the near-point from the far-point creates a normalized 3D Direction Vector.
4. **Plane Intersection:** The mathematical ray `(Origin + t * Direction)` is tested against the physical ground plane `(Z = ground_height)`. The resulting intersection yields the exact `(X, Y, Z)` world coordinate of the user's click.

## 4. 3D Bounding Box Fitting (DBSCAN + PCA)

When the Auto-Annotation pipeline identifies a cluster of 3D points from a 2D camera mask, it must intelligently wrap an Oriented Bounding Box (OBB) around them.



1. **Noise Removal (DBSCAN):** Point clouds are inherently noisy. MSALT first runs Density-Based Spatial Clustering of Applications with Noise (DBSCAN). It finds the densest cluster of points and discards statistical outliers (like floating leaves or sensor ghosting).
2. **Orientation / Heading (PCA):** To determine which way the car or pedestrian is facing, the system projects the clean points onto the 2D ground plane `(X, Y)`. It calculates the covariance matrix of these points and extracts the eigenvectors and eigenvalues (Principal Component Analysis). The eigenvector with the largest eigenvalue represents the longest axis of the object, which becomes the box's primary Heading (Yaw).
3. **Sizing:** The points are rotated to align with this new axis, and the minimum/maximum bounds are calculated to determine the box's exact Length (`dx`), Width (`dy`), and Height (`dz`).

## 5. 3D Box Selection (The Slab Method)

When a user left-clicks the 3D viewer to select an existing bounding box, MSALT uses the **Slab Method** for Ray-OBB Intersection.

Instead of complex polygon collision math, the algorithm transforms the incoming 3D mouse ray into the local coordinate space of the bounding box. It then calculates where the ray enters and exits the three `slabs` (the pairs of parallel planes defining the box's width, length, and height). If the ray successfully enters all three slabs before exiting any of them, a `Hit` is registered.