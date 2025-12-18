from typing import Dict, List

from src.core.objects import BoundingBox3D
import json
import logging
from pathlib import Path
import numpy as np

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
        boxes_dir = Path(boxes_dir)
        meta_dir = Path(meta_dir)

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
