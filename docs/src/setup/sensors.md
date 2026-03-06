# Sensor Setup and Calibration 

- MSALT expects your data to be organized as follows. Define the paths in `config/msalt_setup/docker_setup.yaml` or make your own custom file with the paths
```
paths:
  lidar_folder: "/app/data/lidar"

  cameras:
    - id: "CAM_1"
      name: "Front Center"
      image_folder: "/app/data/camera1"
      intrinsics: "/app/data/camera1_intrinsics.txt"
      extrinsics: "/app/data/camera1_extrinsics.txt"
    - id: "CAM_2"
      name: "Front Left"
      image_folder: "/app/data/camera2"
      intrinsics: "/app/data/camera2_intrinsics.txt"
      extrinsics: "/app/data/camera2_extrinsics.txt"
    - id: "CAM_3"
      name: "Front Right"
      image_folder: "/app/data/camera3"
      intrinsics: "/app/data/camera3_intrinsics.txt"
      extrinsics: "/app/data/camera3_extrinsics.txt"
    - id: "CAM_4"
      name: "Rear Left"
      image_folder: "/app/data/camera4"
      intrinsics: "/app/data/camera4_intrinsics.txt"
      extrinsics: "/app/data/camera4_extrinsics.txt"
    - id: "CAM_5"
      name: "Rear Right"
      image_folder: "/app/data/camera5"
      intrinsics: "/app/data/camera5_intrinsics.txt"
      extrinsics: "/app/data/camera5_extrinsics.txt"

extensions:
  images: ".png"
  lidar: ".pcd"
```
- In `config/config.yaml` make the modification where you have put your desired file setup with the paths
```
defaults:
  - msalt_setup: <custom file setup name here>  
  - models: default
  - _self_
```

## Calibration setup
- for `intrinsics` ensure that your text file structure looks like this
- The matrix is generally formulated as:
  - Focal Length (fx, fy): Represented in pixel units; defines the zoom level.
  - Principal Point (cx, cy): The pixel coordinate where the optical axis intersects the image plane (usually the image center).
```
643.542725 0.000000 646.751343
0.000000 642.712952 362.309418
0.000000 0.000000 1.000000
```
- for `extrinsics` ensure that your text file structure looks like this 
- the calibration results are in this format T_lidar_camera: [x, y, z, qx, qy, qz, qw]
```
0.10336329096238568
-0.019238416589824225
-0.039398644058342445
-0.4230164581873918
0.4368759758405062
-0.5685466143578862
0.5540317726793159s
```
- for `distortion` ensure that your text file structure looks like this
```
-0.055080
0.065999
0.000175
0.000906
-0.021214
```
