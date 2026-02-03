from dataclasses import dataclass, field
from typing import List

@dataclass
class ClassMetrics:
    """
    Accumulator for True Positives, False Positives, and False Negatives.
    Calculates derived metrics (Precision, Recall, F1) on the fly.
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: List[float] = field(default_factory=list)
    
    @property
    def precision(self) -> float:
        """Accuracy of positive predictions."""
        total = self.tp + self.fp
        return self.tp / total if total > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Ability to find all positive instances."""
        total = self.tp + self.fn
        return self.tp / total if total > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """Harmonic mean of Precision and Recall."""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def mean_iou(self) -> float:
        """Average Intersection over Union for True Positives """
        return sum(self.ious) / len(self.ious) if self.ious else 0.0