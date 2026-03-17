# Label Configuration 

## Creating custom classes and directories for saving
- in the `config.yaml` you can put your custom path for saving the annotations in which 3d folder will contain the annotations in json format
- in the `labels` section in the `config.yaml` you can store the type of the classes that you want along with the colour coding for 2D boxes and the points inside the 3D bounding box.
- the ones below are the most common classes in autonomous driving datasets, the naming conventions can be a subject to change.
- also we provide an option for exporting 3D segmentation labels, ensure that you give a unique number mapping as per your choice

```
output:
  dir: "${hydra:runtime.cwd}/MSALT_outputs_annotations_nuscenes"

labels:
  - name: "moving_people"
    color: [255, 0, 0]        # Bright Red
    hotkey: "1"               
    
  - name: "static_people"
    color: [0, 255, 0]        #  Green
    hotkey: "2"

  - name: "unknown"
    color: [0, 255, 0]        # Green
    hotkey: "3"

  - name: "static_car"
    color: [255, 255, 255]    # white
    hotkey: "4"

  - name: "moving_car"
    color: [255, 0, 255]      # Magenta
    hotkey: "5"

  - name: "moving_truck"
    color: [255, 140, 0]      # Dark Orange 
    hotkey: "6"

  - name: "static_truck"
    color: [255, 20, 147]     # Deep Pink 
    hotkey: "7"

  - name: "moving_bus"
    color: [255, 215, 0]      # Gold 
    hotkey: "8"

  - name: "static_bus"
    color: [128, 128, 0]      # Olive (Dark Yellow-Green) 
    hotkey: "9"

  - name: "moving_cyclist"
    color: [0, 255, 255]      # Cyan (Bright Electric Blue)
    hotkey: "10"

  - name: "static_cyclist"
    color: [102, 0, 204]      # Purple
    hotkey: "11"

  - name: "moving_construction_vehicle"
    color: [255, 20, 147]     # Deep Pink 
    hotkey: "12"

  - name: "static_construction_vehicle"
    color: [199, 21, 133]     # Medium Violet Red 
    hotkey: "13"

  - name: "moving_other_vehicle"
    color: [138, 43, 226]     # Blue-Violet (Neon Purple) 
    hotkey: "14"

  - name: "static_other_vehicle"
    color: [75, 0, 130]       # Indigo (Very Dark Purple)
    hotkey: "15"

export:
  segmentation_mapping:
    unknown: 3
    moving_people: 1
    static_people: 2
    moving_car: 5
    static_car: 4
```
