from typing import Dict, List
from src.core.objects import BoundingBox3D
import json
import logging
from pathlib import Path
import numpy as np
from src.core.geometry import GeometryUtils

logger = logging.getLogger(__name__)


class AnnotationManager:
    def __init__(self) -> None:
        # Master storage: Frame Index -> List of Boxes
        self.annotations: Dict[int, List[BoundingBox3D]] = {}
        
        # Store paths for saving later
        self.boxes_dir = None
        self.meta_dir = None

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

    def save_frame(
        self, frame_idx: int, boxes_dir: Path, meta_dir: Path, filename: str
    ):
        boxes_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        boxes = self.get_boxes(frame_idx)

        clean_list = []
        meta_list = []

        for box in boxes:
            clean_struct = {
                "obj_id": str(box.track_id),
                "obj_type": box.label if box.label else "moving_people",
                "psr": {
                    "position": {
                        "x": float(box.x),
                        "y": float(box.y),
                        "z": float(box.z),
                    },
                    "rotation": {"x": 0.0, "y": 0.0, "z": float(box.heading)},
                    "scale": {
                        "x": float(box.dx),
                        "y": float(box.dy),
                        "z": float(box.dz),
                    },
                },
            }
            clean_list.append(clean_struct)

            # We save point_indices as a list of integers to make it JSON serializable
            indices_list = (
                box.point_indices.tolist() if box.point_indices is not None else []
            )

            meta_struct = {
                "obj_id": str(box.track_id),
                "source_2d": box.source_2d,  # {'cam_id':..., 'rect':...}
                "point_indices": indices_list,
            }
            meta_list.append(meta_struct)

            with open(boxes_dir / filename, "w") as f:
                json.dump(clean_list, f, indent=2)

            with open(meta_dir / filename, "w") as f:
                json.dump(meta_list, f)

            logger.info(f"Saved annotations to {filename}")

    def load_frames(self, boxes_dir: Path, meta_dir: Path):
        self.boxes_dir = Path(boxes_dir)
        self.meta_dir = Path(meta_dir)

        if not boxes_dir.exists():
            return

        json_files = sorted(boxes_dir.glob("*.json"))
        count = 0

        for js_file in json_files:
            frame_idx = int(js_file.stem)

            try:
                with open(js_file, "r") as f:
                    clean_data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON: {js_file}")
                continue

            meta_data = {}
            meta_path = meta_dir / js_file.name

            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        raw_meta = json.load(f)
                        for m in raw_meta:
                            meta_data[str(m["obj_id"])] = m
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {js_file.name}: {e}")

            # reconstructing objects
            loaded_boxes = []
            for item in clean_data:
                obj_id = str(item.get("obj_id", -1))
                psr = item["psr"]

                box = BoundingBox3D(
                    x=float(psr["position"]["x"]),
                    y=float(psr["position"]["y"]),
                    z=float(psr["position"]["z"]),
                    dx=float(psr["scale"]["x"]),
                    dy=float(psr["scale"]["y"]),
                    dz=float(psr["scale"]["z"]),
                    heading=float(psr["rotation"].get("z", 0.0)),
                    label=item.get("obj_type", "unknown"),
                    track_id=int(obj_id),
                )

                # merge metadata
                if obj_id in meta_data:
                    m = meta_data[obj_id]
                    box.source_2d = m.get("source_2d")

                    indices = m.get("point_indices")
                    if indices:
                        box.point_indices = np.array(indices, dtype=int)

                loaded_boxes.append(box)

            if loaded_boxes:
                self.annotations[frame_idx] = loaded_boxes
                count += len(loaded_boxes)

        logger.info(f"Loaded {count} boxes from {boxes_dir}")

    def remove_box(self, frame_idx: int, track_id: int):
        if frame_idx not in self.annotations:
            return

        # Filter out the box with the matching ID
        initial_count = len(self.annotations[frame_idx])
        self.annotations[frame_idx] = [
            box for box in self.annotations[frame_idx] 
            if box.track_id != track_id
        ]

        # Only trigger save if something was actually removed
        if len(self.annotations[frame_idx]) < initial_count:
            if self.boxes_dir and self.meta_dir:            
                filename = f"{frame_idx:06d}.json"
                self.save_frame(frame_idx, self.boxes_dir, self.meta_dir, filename=filename)
            else:
                logger.warning("Cannot save deletion: Output directories not set.")
                
    def run_interpolation(self, track_id: int, current_frame_idx: int):
        """
        Finds the previous appearance of track_id and fills frames up to current_frame_idx.
        """
        # searching backwards
        start_frame = -1
        box_start = None
        
        # look back upto 50 frames -> need to check 10
        for f in range(current_frame_idx - 1, max(-1, current_frame_idx - 50),-1):
            frames_boxes = self.get_boxes(f)
            match = next((b for b in frames_boxes if b.track_id == track_id), None)
            if match:
                start_frame = f
                box_start = match
                break
            
        if box_start is None:
            logger.warning(f"No previous frame found for Track ID {track_id}")
            return 0
        
        # Get "End" Keyframe (Current Frame)
        current_boxes = self.get_boxes(current_frame_idx)
        box_end = next((b for b in current_boxes if b.track_id == track_id), None)
        
        if not box_end:
            return 0
        
        total_steps = current_frame_idx - start_frame
        count = 0
        
        for i in range(1, total_steps):
            t = i / total_steps
            target_frame = start_frame + i
            
            # calculating params
            params = GeometryUtils.interpolate_box(box_start, box_end, t)
        
            # create object
            new_box = BoundingBox3D(**params)
            
            # Overwrite existing box if present
            self.remove_box(target_frame, track_id)
            self.add_box(target_frame, new_box)
            
            # Auto-save
            if self.boxes_dir and self.meta_dir:
                filename = f"{target_frame:06d}.json"
                self.save_frame(target_frame, self.boxes_dir, self.meta_dir, filename)
            
            count += 1
            
        return count