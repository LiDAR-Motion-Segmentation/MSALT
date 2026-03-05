# Understanding the Metrics

## Benchmarking
- You can click on `Compare Ground Truth` to see the ground truth bounding boxes in the viewer
- Change the paths in the `benchmark.yaml` config to take in the paths
```
defaults:  
  - _self_  

# Benchmark Settings
output_dir: "/MSALT_outputs_annotations_nuscenes/"                   
data_root: "/home/Downloads/v10-mini"                                       
scene_id: 1                                           
num_frames: 5                                        
iou_threshold: 0.3   # IoU > 0.3 counts as True Positive

# Class Mapping (Grouping diverse labels into Report Categories)
# Format: "Raw_Dataset_Label": "Report_Class_Name"
label_mapping:
  # Cars
  moving_car: "Car"
  static_car: "Car"
  vehicle.car: "Car"
  
  # Pedestrians
  moving_people: "Pedestrian"
  static_people: "Pedestrian"
  human.pedestrian.adult: "Pedestrian"
  human.pedestrian.construction_worker: "Pedestrian"
  human.pedestrian.police_officer: "Pedestrian"
  
  # Large Vehicles
  truck: "Truck"
  vehicle.truck: "Truck"
  bus: "Bus"
  vehicle.bus.rigid: "Bus"
  vehicle.bus.bendy: "Bus"
```
- after this you can run `benchmark/benchmark_nuscenes.py` to generate the results for precision, recall, F1-score and mean IoU for a scene and a series of sequences
```
evaluating 5 frames for Scene 1...

=====================================================================================
  STANDARD IOU BENCHMARK REPORT: SCENE 1  
=====================================================================================
Class           | Precision  | Recall     | F1-Score   | Mean IoU   | Counts (TP/FP/FN) 
-------------------------------------------------------------------------------------
Pedestrian      | 0.82       | 0.84       | 0.83       | 0.88       | 98/21/19          
Car             | 0.89       | 0.89       | 0.89       | 0.96       | 32/4/4            
Truck           | 1.00       | 1.00       | 1.00       | 0.96       | 3/0/0             
-------------------------------------------------------------------------------------
AVERAGE         | -          | -          | 0.91       | 0.93       | -
=====================================================================================

=====================================================================================
  POINT-WISE BENCHMARK REPORT: SCENE 1  
=====================================================================================
Class           | Objects    | GT Points    | Pred Points  | Error/Object   
-------------------------------------------------------------------------------------
Car             | 36         | 383          | 306          | 2.53           
Pedestrian      | 119        | 1170         | 1035         | 1.30           
Truck           | 3          | 4            | 4            | 0.00           
-------------------------------------------------------------------------------------
AVERAGE / TOTAL | 158        | 1557         | 1345         | 1.56           
=====================================================================================
```