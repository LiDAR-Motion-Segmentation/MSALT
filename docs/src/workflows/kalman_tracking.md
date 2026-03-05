# Tracking (Kalman Filter)

The Kalman Filter tracking workflow is a lightweight, purely mathematical approach to forward-predicting an object's trajectory. It is highly effective for objects moving at a constant velocity (like a car driving down a straight highway) or for maintaining a bounding box during brief visual occlusions.

## How to Use It

1. **Select an Object:** In the 3D LiDAR view or Annotation List, select an existing bounding box that you want to track forward.
2. **Trigger Tracking:** Click the **Kalman Filter Tracking (K)** button in the Automation Panel, or press the **`K`** hotkey.
3. **Review:** The system will calculate the object's trajectory and instantly extrapolate the bounding box into future frames.



## Under the Hood

Unlike the SAM2 pipeline, this method does not use neural networks or image processing. 
* MSALT calculates the velocity vector of the selected object based on its movement in previous frames.
* It projects the `X, Y, Z` coordinates and the `Heading` (yaw) forward linearly into the future.
* Because it assumes a constant velocity, the dimensions of the box (`dx, dy, dz`) remain unchanged.

## Configuration

You can control how far into the future the Kalman filter attempts to predict by adjusting the `track_horizon` variable in your `config.yaml`:

```yaml
automation:
  track_horizon: 10  # Number of frames to predict forward
```