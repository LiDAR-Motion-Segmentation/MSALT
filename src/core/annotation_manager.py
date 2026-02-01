from typing import Dict, List
from src.core.objects import BoundingBox3D
import json
import logging
from pathlib import Path
import numpy as np
from copy import deepcopy
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
                "visual_overrides": box.visual_overrides
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
                    
                    if "visual_overrides" in m:
                        box.visual_overrides = m["visual_overrides"]
                    elif "visual_override_2d" in m:
                        box.visual_overrides = m["visual_override_2d"]
                    
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
                
    def run_smart_interpolation(self, 
                                track_id: int, 
                                start_frame, 
                                end_frame, 
                                data_controller, 
                                seg_engine) -> int:
        """
        Finds the previous appearance of track_id and fills frames up to current_frame_idx.
        """
        logger.info(f"Starting Track ID {track_id}")
        count = 0
        boxes = self.get_boxes(start_frame)
        current_box = next((b for b in boxes if b.track_id == track_id), None)
        if not current_box:
            logger.warning(f"Track ID {track_id} not found in start frame.")
            return 0
        
        for f_idx in range(start_frame + 1, end_frame + 1):
            frame_data = data_controller.get(f_idx)
            if not frame_data or frame_data.point_cloud is None:
                break
            
            points = frame_data.point_cloud
            new_box = None
            best_cam_id = None
            rect_2d = None    
            
            if frame_data.images:
                for cam_id, image in frame_data.images.items():
                    calib = data_controller.get_calibration(cam_id)
                    if not calib:
                        continue
                    
                    # project box -> image
                    candidate_rect = GeometryUtils.project_box_to_image(
                        current_box, calib['extrinsic'], calib['intrinsic'], image.shape
                    )    
                
                    if candidate_rect:
                        best_cam_id = cam_id
                        rect_2d = candidate_rect
                        logger.debug(f"Frame {f_idx}: Object found in {cam_id}")
                        break # Found it!
                
            if best_cam_id and rect_2d:        
                image = frame_data.images[cam_id]
                calib = data_controller.get_calibration(cam_id)
                if not calib:
                    logger.warning(f"Frame {f_idx}: No calibration found for {cam_id}")
                    break
                
                if not calib: 
                    break
            
                # SAM2 inference
                mask = seg_engine.get_mask_from_box(image, rect_2d)
                if mask is not None: 
                    logger.warning(f"Frame {f_idx}: SAM returned no mask")
            
                    uv, valid = GeometryUtils.project_lidar_to_image(points, calib['extrinsic'], calib['intrinsic'])
                        
                    if hasattr(mask, 'cpu'): 
                        mask = mask.cpu().numpy().astype(bool)
                    
                    # Get points strictly on the object mask
                    h, w = mask.shape[:2]
                    valid_indices = np.where(valid)[0]
                    uv_valid = uv[valid_indices].astype(int)
                    
                    in_bounds = (uv_valid[:,0] >= 0) & (uv_valid[:,0] < w) & \
                                (uv_valid[:,1] >= 0) & (uv_valid[:,1] < h)
                    
                    candidate_indices = valid_indices[in_bounds]
                    candidate_uv = uv_valid[in_bounds]
                    
                    on_mask = mask[candidate_uv[:, 1], candidate_uv[:, 0]]
                    object_indices = candidate_indices[on_mask]
                    
                    if len(object_indices) >= 10:
                        object_points = points[object_indices]
                        new_box = GeometryUtils.fit_box_with_pca(object_points, current_box)
                        if new_box:
                            logger.info(f"Frame {f_idx}: Tracked via SAM ({best_cam_id})")
                        else:
                            logger.warning(f"Frame {f_idx}: Only found {len(object_indices)} points on mask (Too few)") 
            
            # lidar tracking
            if new_box is None:
                logger.info(f"Frame {f_idx}: Fallback to LiDAR Tracking (Object off-screen)")
                indices = GeometryUtils.get_points_in_box(points, current_box)
                
                # Better: Simple distance check if get_points_in_box is strict
                search_box = deepcopy(current_box)
                search_box.dx += 1.0
                search_box.dy += 1.0
                
                logger.info(f"Frame {f_idx}: Checking LiDAR points in area {search_box.dx:.2f}x{search_box.dy:.2f}")
                indices = GeometryUtils.get_points_in_box(points, search_box)
                logger.info(f"Frame {f_idx}: Found {len(indices)} points in search region.")
                
                if len(indices) > 10:
                    cloud_subset = points[indices]
                    
                    # Fit
                    logger.info(f"Frame {f_idx}: Running PCA Fit on {len(cloud_subset)} points...")
                    try:
                        new_box = GeometryUtils.fit_box_with_pca(cloud_subset, current_box)
                        if new_box:
                            logger.info(f"Frame {f_idx}: LiDAR Fit Successful!")
                        else:
                            logger.warning(f"Frame {f_idx}: LiDAR Fit returned None (DBSCAN noise?).")
                    except Exception as e:
                        logger.error(f"Frame {f_idx}: PCA Crash: {e}")
                else:
                    logger.warning(f"Frame {f_idx}: Too few points ({len(indices)}) to fit box.")
            
            if new_box:
                new_box.track_id = track_id
                self.add_box(f_idx, new_box)
                current_box = new_box
                count += 1
            else:
                logger.warning(f"Frame {f_idx}: Tracking Lost completely.")
                break
                
        return count
    
    def deselect_all(self):
        """
        Iterates through ALL boxes in ALL frames and sets selected = False.
        """
        for frame_idx, boxes in self.annotations.items():
            for box in boxes:
                box.selected = False