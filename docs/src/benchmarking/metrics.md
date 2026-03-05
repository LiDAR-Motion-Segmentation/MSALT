# Understanding the Metrics

MSALT includes a highly detailed, built-in benchmarking suite to evaluate the accuracy of your auto-generated 3D bounding boxes against established datasets like NuScenes. 

Unlike standard autonomous driving evaluators that only look at geometric overlap, MSALT also performs **Point-Level Tracking** to tell you exactly how well your bounding boxes encapsulate the raw LiDAR data.

## Running the Evaluator

The benchmarking suite is powered by Hydra and can be run directly from the command line. It compares your saved JSON annotations against the ground truth metadata.

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

## Column Breakdown
1. Geometric Metrics (Box Accuracy)
- `GT & Pred`: The total number of Ground Truth boxes vs. Predicted (MSALT) boxes in the current frame.

- `Best IoU`: Intersection over Union. This measures the Bird's-Eye-View (BEV) 2D overlap between your predicted box and the ground truth box. A score of 1.0 is a perfect match.

- `Status`: *TP (True Positive)*: The IoU was higher than the configured threshold (default 0.25).

- `FP (False Positive)`: A box was predicted, but it didn't overlap sufficiently with any ground truth box.

- `Missed (FN / False Negative)`: A ground truth object existed, but the AI failed to predict a box for it.

2. Point-Level Metrics (Depth & Fit Accuracy)
- `GT Pts`: The total number of raw LiDAR points sitting strictly inside the Ground Truth bounding boxes.

- `Pred Pts`: The total number of LiDAR points sitting inside your MSALT-generated boxes.

- `Err/Obj (Error per Object)`: The absolute difference between GT points and Pred points, divided by the number of objects.

**Why this matters** : If your IoU is high (e.g., 0.80), but your Err/Obj is also very high, it means your box is placed in the correct X/Y location, but is likely too tall (capturing tree branches) or sunk too low (capturing the ground plane).

## The Global Summary Report
After processing all frames, the evaluator prints a global summary report for each class.

- `Precision`: TP / (TP + FP). Answers the question: "Out of all the boxes MSALT auto-generated, how many were actually correct?"

- `Recall`: TP / (TP + FN). Answers the question: "Out of all the real cars/pedestrians in the scene, how many did MSALT successfully find?"

- `F1 Score`: The harmonic mean of Precision and Recall. This is your ultimate single-number score for the pipeline's performance.

## Configuration & Distance Filtering
- Autonomous driving sensors suffer from point sparsity at long distances. To benchmark strictly in the "safety-critical zone," MSALT allows you to set a max_radius.

- In your `benchmark_nuscenes.yaml` config, you can define:

```
max_radius: 20.0
iou_threshold: 0.25
```

- `Setting max_radius`: 20.0 tells the evaluator to mathematically ignore any ground truth or predicted boxes that are further than 20 meters from the ego-vehicle. This ensures your precision and recall scores are not penalized by YOLO failing to detect cars that are 80 meters away and only consist of 2 LiDAR points.