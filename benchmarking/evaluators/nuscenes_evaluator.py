import json
import logging
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
        self.metrics = defaultdict(ClassMetrics)
        
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
                    'dx': p['scale']['x'],
                    'dy': p['scale']['y'],
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
                'dx': box.dx, # In your system, dx is length/scale_x
                'dy': box.dy, # dy is width/scale_y
                'heading': box.heading,
                'label': unified
            })
        return standardized
    
    def run(self):
        self.load_ground_truth()
        limit = min(len(self.loader), self.cfg.num_frames)
        user_dir = Path(self.cfg.output_dir) / "3d"
        
        print(f"\nEvaluating {limit} frames for Scene {self.cfg.scene_id}...")
        print(f"{'Frame':<6} | {'Class':<10} | {'GT':<3} | {'Pred':<4} | {'Best IoU':<8} | {'Status'}")
        print("-" * 65)
        
        for i in range(limit):
            frame = self.loader.get(i)
            gt_std = self._parse_gt_box(frame.metadata.get('gt_boxes', []))
            pred_std = self._parse_user_json(user_dir / f"{i:06d}.json")
            
            all_labels = set([b['label'] for b in gt_std] + [b['label'] for b in pred_std])
            
            for label in all_labels:
                cls_gt = [b for b in gt_std if b['label'] == label]
                cls_pred = [b for b in pred_std if b['label'] == label]
                self._match_and_update(i, label, cls_gt, cls_pred)
                
    def _match_and_update(self, frame_idx, label, gt_list, pred_list):
        metrics = self.metrics[label]
        matched_gt = set()
        
        for p_box in pred_list:
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
                
            print(f"{frame_idx:<6} | {label:<10} | {len(gt_list):<3} | {len(pred_list):<4} | {best_iou:.2f}     | {status}")
            
        fn = len(gt_list) - len(matched_gt)
        metrics.fn += fn
        if fn > 0:
            print(f"{frame_idx:<6} | {label:<10} | {len(gt_list):<3} | -    | -        | Missed {fn}")
            
    def print_report(self):
        print("\n" + "="*85)
        print(f"  BENCHMARK REPORT: SCENE {self.cfg.scene_id}  ")
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