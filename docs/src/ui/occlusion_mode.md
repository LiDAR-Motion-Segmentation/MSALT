## Occlusion Mode

This document explains the math used in `_handle_camera_hover` to back‑project a 2D image pixel into a 3D ray in LiDAR space and pick an occlusion‑aware hit point on the point cloud.

![hover](../assets/occulsion_mode.png)

### 1. From pixel to camera‑frame ray

Given:
- Pixel coordinates (u, v)
- Camera intrinsic matrix K

We first build a ray in the camera frame

This yields an (unnormalized) 3D direction vector in camera coordinates.

### 2. Camera ray to LiDAR frame

Let the camera→LiDAR pose be


where:
- R is the rotation from camera frame to LiDAR frame
- t is the camera origin expressed in LiDAR coordinates

<!-- Then:

- Ray origin in LiDAR:
\[
\text{origin}_\text{lidar} = t
\]

- Ray direction in LiDAR:
\[
\tilde{r}_\text{lidar} = R \,\text{ray}_\text{cam}
\]
\[
\text{ray}_\text{lidar} =
\frac{\tilde{r}_\text{lidar}}{\left\lVert \tilde{r}_\text{lidar} \right\rVert}
\] -->

### 3. Occlusion‑aware hit point on the point cloud

1. **Vector from camera origin to each point**
<!-- \[
v_i = P_i - \text{origin}_\text{lidar}
\] -->

2. **Depth along the ray**
<!-- \[
t_i = v_i \cdot \text{ray}_\text{lidar}
\] -->

We ignore points behind the camera
<!-- \[
t_i > 0
\] -->

3. **Perpendicular distance to the ray**

<!-- Project \(v_i\) onto the ray:
\[
\text{proj}_i = t_i \,\text{ray}_\text{lidar}
\]

Rejection (component orthogonal to the ray) and its norm:
\[
r_i = v_i - \text{proj}_i
\]
\[
d_i = \left\lVert r_i \right\rVert
\] -->

4. **Choose the visible hit**

Define a small distance threshold (e.g. (0.3) m).  
We consider only points that the ray passes “through”:

<!-- \[
d_i < \varepsilon
\] -->

Among these, the visible surface point is the one closest along the ray (smallest positive depth):

<!-- \[
i^\* = \arg\min_{i} \{\, t_i \mid t_i > 0,\ d_i < \varepsilon \,\}
\]

If such an index \(i^\*\) exists, the occlusion‑aware hit point is
\[
\text{hit\_point} = P_{i^\*}.
\] -->
Otherwise, no hit point is reported (only the infinite ray is drawn).

### 4. Rendering in the LiDAR view

The UI passes:
- `origin = cam_origin_lidar = origin_lidar`
- `direction = ray_lidar`
- `hit_point = P_i` or `None`

to:

```python
self.lidar_widget.update_laser_pointer(origin, direction, hit_point)
```

which draws a long red ray and, when available, a red blob at the selected 3D hit point on the point cloud.

# Occlusion Mode

- (closest along the ray) as the true occlusion-aware hit.
1. Send to the LiDAR viewer
2. Use the camera-origin ray and the chosen hit
3. origin = cam_origin_lidar 
4. direction = ray_lidar
5. hit_point = selected 3D point or None
6. Pass them to `self.lidar_widget.update_laser_pointer(origin, direction, hit_point)`.
