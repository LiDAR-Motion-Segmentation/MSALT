from typing import Dict, List, Optional

from numpy import delete
from src.core.objects import BoundingBox3D
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AnnotationManager:
    def __init__(self) -> None:
        # Master storage: Frame Index -> List of Boxes
        self.annotations: Dict[int, List[BoundingBox3D]] = {}

    def get_boxes(self, frame_idx: int):
        return self.annotations.get(frame_idx, [])

    def add_box(self, frame_idx: int, box: BoundingBox3D):
        if frame_idx not in self.annotations:
            self.annotations[frame_idx] = []
            
        # Simple ID assignment if not tracked (1, 2, 3...)
        if box.track_id == -1:
            box.track_id = len(self.annotations[frame_idx]) + 1
            
        self.annotations[frame_idx].append(box)

    def delete_box(self, frame_idx: int, box: BoundingBox3D):
        if frame_idx in self.annotations:
            if box in self.annotations[frame_idx]:
                self.annotations[frame_idx].remove(box)

    def save_frame_json(self, frame_idx: int, output_dir: Path, filename: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        boxes = self.get_boxes(frame_idx)
        json_list = []
        
        for box in boxes:
            obj_struct = {
                "obj_id": str(box.track_id),
                "obj_type": box.label if box.label else "moving_people",
                "psr": {
                    "position": {
                        "x": float(box.x),
                        "y": float(box.y),
                        "z": float(box.z)
                    },
                    "rotation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": float(box.heading)
                    },
                    "scale": {
                        "x": float(box.dx),
                        "y": float(box.dy),
                        "z": float(box.dz)
                    }
                }
            }
            json_list.append(obj_struct)
            
            file_path = output_dir / filename
            with open(file_path, 'w') as f:
                json.dump(json_list, f, indent=2)
                
            logger.info(f"Saved annotations to {file_path}")