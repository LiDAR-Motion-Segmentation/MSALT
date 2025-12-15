import torch
import numpy as np
from ultralytics import SAM
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class SegmentationEngine:
    def __init__(self, model_cfg: DictConfig) -> None:
        """
        Initializes the engine using Hydra configuration.
        
        Args:
            model_cfg (DictConfig): The 'models' section from config.yaml
                                    (e.g., cfg.models)
        """
        self.cfg = model_cfg
        self.model = None
        # print(f"DEBUG: Received Config keys: {model_cfg.keys()}")

        req_device = self.cfg.segmentation.device
        if req_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Config requested CUDA but not available. Falling back to CPU."
            )
            self.device = "cpu"
        else:
            self.device = req_device

    def _ensure_loaded(self):
        if self.model is None:
            weights = self.cfg.segmentation.weights
            logger.info(f"Loading {self.cfg.segmentation.name} weights: {weights}")

            try:
                self.model = SAM(weights)
                logger.info(f"Model loaded successfully on {self.device}")
            except Exception as e:
                logger.critical(f"Failed to load model weights: {e}")
                raise e

    def get_mask_from_box(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Args:
            image: RGB numpy array
            bbox: [x, y, w, h]
        """
        self._ensure_loaded()
        x, y, w, h = bbox
        box_xyxy = [x, y, x + w, y + h]

        # Run inference using the config parameters
        results = self.model(
            image,
            bboxes=[box_xyxy],
            verbose=False,
            device=self.device,
            conf=self.cfg.segmentation.conf_threshold,
        )

        if results and results[0].masks is not None:
            mask_tensor = results[0].masks.data[0]
            return mask_tensor.cpu().numpy().astype(bool)

        return np.zeros(image.shape[:2], dtype=bool)
