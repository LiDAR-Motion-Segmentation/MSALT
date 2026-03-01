import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List
from omegaconf import DictConfig

from benchmarking.evaluators.base_evaluator import BaseEvaluator
from benchmarking.metrics import ClassMetrics
from benchmarking.geometry import GeometryUtils

from src.data.structures import SensorConfig
from src.data.loaders.nuscenes_loader import NuScenesLoader

class NuScenesEvaluator(BaseEvaluator):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.metrics: dict = defaultdict(ClassMetrics)
        
        # Track point-wise metrics alongside standard metrics
        self.point_metrics: dict = defaultdict(lambda: {'gt_pts': 0, 'pred_pts': 0, 'abs_error': 0, 'objs': 0})
        
        # safely fetch max_radius from the Hydra config
        self.max_radius = self.cfg.get('max_radius', None)

        try:
            self.loader_cls = NuScenesLoader
            self.sensor_cfg_cls = SensorConfig
        except Exception as e:
            logging.error(f"Could not import MSALT core modules: {e}")
            exit(1)
            
    def load_ground_truth(self):
        logging.info(f"Initializing NuScenes Loader for Scene {self.cfg.scene_id}...")
        
        # Hydra converts paths to strings, ensure Path object if needed by loader
        cfg = self.sensor_cfg_cls(
            lidar_path=Path(self.cfg.data_root),
            cameras=[],
            extra_params={"version": "v1.0-mini", "scenes": [self.cfg.scene_id]}
        )
        self.loader = self.loader_cls(cfg)
        
    def _get_unified_label(self, raw_label: str) -> str:
        mapping = self.cfg.label_mapping
        return mapping.get(raw_label, None)
    
    def _parse_user_json(self, json_path: Path) -> List[dict]:
        if not json_path.exists():
            return []

        standardized = []
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data:
                raw_label = item.get('obj_type', 'unknown')
                
                unified = self._get_unified_label(raw_label)
                if not unified:
                    continue
                
                p = item['psr']
                standardized.append({
                    'x': p['position']['x'],
                    'y': p['position']['y'],
                    'z': p['position']['z'],
                    'dx': p['scale']['x'],
                    'dy': p['scale']['y'],
                    'dz': p['scale']['z'], 
                    'heading': p['rotation']['z'],
                    'label': unified
                })
                
        return standardized
    
    def _parse_gt_box(self, gt_boxes) -> List[dict]:
        standardized = []
        for box in gt_boxes:
            unified = self._get_unified_label(box.label)
            if not unified:
                continue
            
            standardized.append({
                'x': box.x,
                'y': box.y,
                'z': box.z,
                'dx': box.dx, # In your system, dx is length/scale_x
                'dy': box.dy, # dy is width/scale_y
                'dz': box.dz,
                'heading': box.heading,
                'label': unified
            })
        return standardized
    
    def _filter_by_radius(self, boxes: List[dict]) -> List[dict]:
        """Filters out boxes that are further than max_radius from the ego vehicle (0,0)."""
        if self.max_radius is None:
            return boxes
        
        filtered_boxes = []
        for b in boxes:
            # Calculate 2D Euclidean distance from the origin (ego vehicle)
            dist = np.hypot(b['x'], b['y'])
            if dist <= self.max_radius:
                filtered_boxes.append(b)
        return filtered_boxes
    
    def _get_points_in_box(self, points: np.ndarray, box: dict) -> int:
        """Fast 3D point counting using AABB."""
        if points is None or len(points) == 0:
            return 0
            
        cx, cy, cz = box['x'], box['y'], box['z']
        w, l, h = box['dx'], box['dy'], box['dz']
        yaw = box['heading']
        
        # Translate
        translated = points[:, :3] - np.array([cx, cy, cz])
        
        # Rotate (Inverse Yaw)
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        rot_mat = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [    0,      0, 1]
        ])
        rotated = np.dot(translated, rot_mat.T)
        
        # AABB Filter
        mask_x = np.abs(rotated[:, 0]) <= (w / 2.0)
        mask_y = np.abs(rotated[:, 1]) <= (l / 2.0)
        mask_z = np.abs(rotated[:, 2]) <= (h / 2.0)
        
        return int(np.sum(mask_x & mask_y & mask_z))
    
    def run(self):
        self.load_ground_truth()
        limit = min(len(self.loader), self.cfg.num_frames)
        user_dir = Path(self.cfg.output_dir) / "3d"
        
        # Ensure radius filter text is printed if it exists
        radius_text = f" (Radius: <= {self.max_radius}m)" if getattr(self, 'max_radius', None) else " (All Distances)"
        print(f"\nEvaluating {limit} frames for Scene {self.cfg.scene_id}{radius_text}...")
       
        print(f"{'Frame':<6} | {'Class':<10} | {'GT':<3} | {'Pred':<4} | {'Best IoU':<8} | {'Status':<10} | {'GT Pts':<8} | {'Pred Pts':<8} | {'Err/Obj'}")
        print("-" * 95)
        
        for i in range(limit):
            frame = self.loader.get(i)
            
            # Parse boxes
            gt_std = self._parse_gt_box(frame.metadata.get('gt_boxes', []))
            pred_std = self._parse_user_json(user_dir / f"{i:06d}.json")
            
            # Apply Distance Filter 
            if hasattr(self, '_filter_by_radius'):
                gt_std = self._filter_by_radius(gt_std)
                pred_std = self._filter_by_radius(pred_std)
            
            points = getattr(frame, 'point_cloud', np.array([]))
            
            frame_gt_pts = defaultdict(int)
            frame_pred_pts = defaultdict(int)
            frame_gt_objs = defaultdict(int)
            frame_pred_objs = defaultdict(int)
            
            for box in gt_std:
                pts = self._get_points_in_box(points, box)
                lbl = box['label']
                frame_gt_pts[lbl] += pts
                frame_gt_objs[lbl] += 1
                
            for box in pred_std:
                pts = self._get_points_in_box(points, box)
                lbl = box['label']
                frame_pred_pts[lbl] += pts
                frame_pred_objs[lbl] += 1
            
            all_labels = set([b['label'] for b in gt_std] + [b['label'] for b in pred_std])
            
            # Update global point metrics
            for label in all_labels:
                c_gt_p = frame_gt_pts[label]
                c_pr_p = frame_pred_pts[label]
                c_err = abs(c_gt_p - c_pr_p)
                c_objs = max(frame_gt_objs[label], frame_pred_objs[label])
                
                self.point_metrics[label]['gt_pts'] += c_gt_p
                self.point_metrics[label]['pred_pts'] += c_pr_p
                self.point_metrics[label]['abs_error'] += c_err
                self.point_metrics[label]['objs'] += c_objs
            
            for label in all_labels:
                cls_gt = [b for b in gt_std if b['label'] == label]
                cls_pred = [b for b in pred_std if b['label'] == label]
                
                c_gt_p = frame_gt_pts[label]
                c_pr_p = frame_pred_pts[label]
                c_objs = max(frame_gt_objs[label], frame_pred_objs[label])
                
                self._match_and_update(i, label, cls_gt, cls_pred, c_gt_p, c_pr_p, c_objs)
                
    def _match_and_update(self, frame_idx, label, gt_list, pred_list, gt_pts, pred_pts, objs):
        metrics = self.metrics[label]
        matched_gt = set()
        
        # Calculate Error per Object for this specific frame
        err_per_obj = (abs(gt_pts - pred_pts) / objs) if objs > 0 else 0.0
        
        # Edge Case: Ground Truth exists, but no predictions were made
        if not pred_list:
            fn = len(gt_list)
            metrics.fn += fn
            if fn > 0:
                print(f"{frame_idx:<6} | {label:<10} | {len(gt_list):<3} | -    | -        | {'Missed ' + str(fn):<10} | {gt_pts:<8} | {pred_pts:<8} | {err_per_obj:<7.2f}")
            return

        for p_idx, p_box in enumerate(pred_list):
            best_iou = 0.0
            best_gt_idx = -1
            
            for g_idx, g_box in enumerate(gt_list):
                if g_idx in matched_gt:
                    continue
                iou = GeometryUtils.calculate_bev_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
                    
            status = "FP"
            if best_iou >= self.cfg.iou_threshold:
                metrics.tp += 1
                metrics.ious.append(best_iou)
                matched_gt.add(best_gt_idx)
                status = "TP"
            else:
                metrics.fp += 1
                
            # We only print the point stats on the FIRST line for each class to keep the console clean
            if p_idx == 0:
                print(f"{frame_idx:<6} | {label:<10} | {len(gt_list):<3} | {len(pred_list):<4} | {best_iou:<8.2f} | {status:<10} | {gt_pts:<8} | {pred_pts:<8} | {err_per_obj:<7.2f}")
            else:
                print(f"{frame_idx:<6} | {label:<10} | {len(gt_list):<3} | {len(pred_list):<4} | {best_iou:<8.2f} | {status:<10} | {'-':<8} | {'-':<8} | {'-':<7}")
            
        fn = len(gt_list) - len(matched_gt)
        metrics.fn += fn
        if fn > 0:
            print(f"{frame_idx:<6} | {label:<10} | {len(gt_list):<3} | -    | -        | {'Missed ' + str(fn):<10} | {'-':<8} | {'-':<8} | {'-':<7}")
            
    def print_report(self):
        print("\n" + "="*85)
        print(f"  STANDARD IOU BENCHMARK REPORT: SCENE {self.cfg.scene_id}  ")
        print("="*85)
        
        headers = f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Mean IoU':<10} | {'Counts (TP/FP/FN)':<18}"
        print(headers)
        print("-" * 85)
        
        f1s, ious = [], []
        
        for label, m in self.metrics.items():
            stats = f"{m.tp}/{m.fp}/{m.fn}"
            print(f"{label:<15} | {m.precision:.2f}       | {m.recall:.2f}       | {m.f1_score:.2f}       | {m.mean_iou:.2f}       | {stats:<18}")
            f1s.append(m.f1_score)
            ious.append(m.mean_iou)
            
        print("-" * 85)
        if f1s:
            print(f"{'AVERAGE':<15} | {'-':<10} | {'-':<10} | {sum(f1s)/len(f1s):.2f}       | {sum(ious)/len(ious):.2f}       | -")
        print("="*85 + "\n")

        # Print Point-Wise Report
        print("="*85)
        print(f"  POINT-WISE BENCHMARK REPORT: SCENE {self.cfg.scene_id}  ")
        print("="*85)
        print(f"{'Class':<15} | {'Objects':<10} | {'GT Points':<12} | {'Pred Points':<12} | {'Error/Object':<15}")
        print("-" * 85)
        
        global_gt = global_pred = global_err = global_objs = 0
        
        for label in sorted(self.point_metrics.keys()):
            m = self.point_metrics[label]
            objs = m['objs']
            err_per_obj = (m['abs_error'] / objs) if objs > 0 else 0.0
            
            global_gt += m['gt_pts']
            global_pred += m['pred_pts']
            global_err += m['abs_error']
            global_objs += objs
            
            print(f"{label:<15} | {objs:<10} | {m['gt_pts']:<12} | {m['pred_pts']:<12} | {err_per_obj:<15.2f}")
            
        print("-" * 85)
        global_err_per_obj = (global_err / global_objs) if global_objs > 0 else 0.0
        print(f"{'AVERAGE / TOTAL':<15} | {global_objs:<10} | {global_gt:<12} | {global_pred:<12} | {global_err_per_obj:<15.2f}")
        print("="*85 + "\n")