import logging
from typing import List, Dict, Set
from src.core.objects import BoundingBox3D

logger = logging.getLogger(__name__)

class OntologyValidator:
    def __init__(self, label_config: List[Dict]) -> None:
        """
        Initializes the validator with the strict label schema.
        Assumes label_config is a list of dicts: [{'name': 'car', ...}, ...]
        """
        self.allowed_labels: Set[str] = {
            item['name'] for item in label_config if 'name' in item
        }
        
    def validate_label(self, label: str) -> bool:
        """Returns True if the label exactly matches an allowed ontology class."""
        return label in self.allowed_labels
    
    def audit_dataset(self, annotations: Dict[int, List[BoundingBox3D]]) -> List[str]:
        """
        Scans all loaded annotations and returns a list of critical error messages.
        """
        errors = []
        id_to_class_map = {} # Tracks the FIRST seen class for every track_id
        
        for frame_idx, boxes in annotations.items():
            for box in boxes:
                # STRICT ONTOLOGY CHECK
                if not self.validate_label(box.label):
                    errors.append(
                        f"Frame {frame_idx:04d} | ID {box.track_id}: "
                        f"Unknown label '{box.label}'. "
                        f"Allowed: {', '.join(self.allowed_labels)}"
                    )
                
                # TEMPORAL ID CONSISTENCY CHECK
                if box.track_id in id_to_class_map:
                    original_class = id_to_class_map[box.track_id]
                    if original_class != box.label:
                        errors.append(
                            f"Frame {frame_idx:04d} | ID {box.track_id}: "
                            f"Class inconsistency! ID started as '{original_class}' "
                            f"but swapped to '{box.label}'."
                        )
                else:
                    # Register the ID and its class the first time we see it
                    id_to_class_map[box.track_id] = box.label
                    
        return errors