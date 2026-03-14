import torch
import numpy as np
from ultralytics import SAM, YOLO
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
        self.model = None  # SAM Model
        self.yolo_model = None  # YOLO Model

        req_device = self.cfg.segmentation.device
        if req_device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Config requested CUDA but not available. Falling back to CPU"
            )
            self.device = "cpu"
        else:
            logger.info("CUDA available using GPU")
            self.device = req_device

    def _ensure_loaded(self):
        # load SAM2
        if self.model is None:
            weights = self.cfg.segmentation.weights
            logger.info(f"Loading {self.cfg.segmentation.name} weights: {weights}")

            try:
                self.model = SAM(weights)
                logger.info(f"Model loaded successfully on {self.device}")
            except Exception as e:
                logger.critical(f"Failed to load model weights: {e}")
                raise e

    def _ensure_yolo_loaded(self):
        # Load YOLO
        if self.yolo_model is None:
            weights = getattr(self.cfg.detection, "weights")
            logger.info(f"Loading YOLO weights: {weights}")
            try:
                self.yolo_model = YOLO(weights)
                logger.info(f"YOLO loaded successfully on {self.device}")
            except Exception as e:
                logger.critical(f"Failed to load YOLO weights: {e}")
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

        if self.model is None:
            return None

        # Run inference using the config parameters
        try:
            results = self.model(
                image,
                bboxes=[box_xyxy],
                verbose=False,
                device=self.device,
                conf=self.cfg.segmentation.conf_threshold,
            )

            # Check if we got any results
            if not results:
                return None

            # Check if we got any masks
            if results[0].masks is None or len(results[0].masks.data) == 0:
                return None

            # Usually we take the first mask (highest confidence)
            mask_tensor = results[0].masks.data[0]
            return mask_tensor.cpu().numpy().astype(bool)

        except Exception as e:
            # Catch inference errors safely
            print(f"SAM Inference Error: {e}")
            return None

    def get_yolo_detection(self, image: np.ndarray) -> list:
        """Runs YOLO and returns a list of dictionaries with box, class_id, and conf."""
        self._ensure_yolo_loaded()
        if self.yolo_model is None:
            return []

        conf_thresh = getattr(self.cfg.detection, "conf_threshold", 0.4)
        # Standard COCO: 0=Person, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        allowed_classes = getattr(self.cfg.detection, "classes", [0, 2, 3, 5, 7])

        try:
            results = self.yolo_model(
                image,
                device=self.device,
                verbose=False,
                conf=conf_thresh,
                classes=allowed_classes,
            )

            detections = []

            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    # YOLO format is center_x, center_y, width, height
                    cx, cy, w, h = box.xywh[0].cpu().numpy()

                    # Convert to top-left x, y for MSALT standard
                    x = cx - w / 2
                    y = cy - h / 2
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    detections.append(
                        {
                            "box": [int(x), int(y), int(w), int(h)],
                            "class_id": cls_id,
                            "conf": conf,
                        }
                    )

            return detections
        except Exception as e:
            logger.error(f"YOLO Inference Error: {e}")
            return []
