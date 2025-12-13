from typing import Dict, List, Optional

from numpy import delete
from .objects import BoundingBox3D

class AnnotationManager:
    def __init__(self) -> None:
        # Master storage: Frame Index -> List of Boxes
        self._storage: Dict[int, List[BoundingBox3D]] = {}
        
        # Clipboard for copy/paste
        self._clipboard: Optional[BoundingBox3D] = None
        
    def get_boxes(self, frame_idx: int):
        return self._storage.get(frame_idx, [])
    
    def add_box(self, frame_idx: int, box: BoundingBox3D):
        if frame_idx not in self._storage:
            self._storage[frame_idx] = []
        self._storage[frame_idx].append(box)
        
    def delete_selected(self, frame_idx: int):
        if frame_idx in self._storage:
            # Keep only unselected boxes
            self._storage[frame_idx] = [b for b in self._storage[frame_idx] if not b.selected]
            
    def clear_selection(self, frame_idx: int):
        for box in self.get_boxes(frame_idx):
            box.selected = False