"""
Object detection module using YOLOv11.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from typing import List, Tuple, Dict, Any
from loguru import logger

from config import Config

class BaseObjectDetector:
    """Base class for YOLO-based object detectors."""
    
    def __init__(self):
        """Initialize the object detector."""
        self.model = None
        self.device = self._setup_device()
        self.class_names: Dict[int, str] = {}
        self.model_name = ""
        self.model_type = ""
        self._load_model()
    
    def get_model_info(self) -> Tuple[str, str]:
        return self.model_name, self.model_type
    
    def _setup_device(self) -> str:
        """Setup the computation device (GPU/CPU)."""
        if Config.USE_GPU and Config.validate_gpu_availability():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def _load_model(self) -> None:
        """Load the YOLO model."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class info
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=Config.MODEL_CONFIDENCE,
                iou=Config.MODEL_IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    # Handle both tensor and numpy array cases
                    boxes = result.boxes.xyxy
                    confidences = result.boxes.conf
                    class_ids = result.boxes.cls
                    
                    # Convert to numpy arrays safely
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    if isinstance(confidences, torch.Tensor):
                        confidences = confidences.cpu().numpy()
                    if isinstance(class_ids, torch.Tensor):
                        class_ids = class_ids.cpu().numpy()
                    
                    # Ensure class_ids are integers
                    class_ids = class_ids.astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().numpy()
                        x1, y1, x2, y2 = box.astype(int)
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names.get(class_id, f'Class_{class_id}')
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                       camera_id: int = 0) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            camera_id: Camera identifier for color selection
            
        Returns:
            Frame with drawn detections
        """
        if not detections:
            return frame
        
        # Select color based on camera ID
        color = Config.BBOX_COLORS[camera_id % len(Config.BBOX_COLORS)]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - baseline - 10),
                (x1 + label_width, y1),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame


class ObjectDetector(BaseObjectDetector):
    """YOLOv11-based object detector that can handle both PyTorch and TensorRT models."""

    def _load_model(self) -> None:
        """Load the YOLOv11 model, handling both .pt and .engine files."""
        model_path = Config.get_full_model_name()
        self.model_name = os.path.basename(model_path)
        try:
            if model_path.endswith(".engine"):
                self._load_model_with_fallback(model_path, is_tensorrt=True)
            else:
                self._load_model_with_fallback(model_path, is_tensorrt=False)

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _load_model_with_fallback(self, model_path: str, is_tensorrt: bool = False) -> None:
        """Load a model with automatic fallback support."""
        try:
            if is_tensorrt:
                # Handle TensorRT-specific logic
                if not os.path.exists(model_path):
                    logger.warning(f"TensorRT engine not found at {model_path}")
                    logger.info("Attempting to export PyTorch model to TensorRT...")
                    self._export_to_tensorrt(model_path)

                logger.info(f"Loading TensorRT model from {model_path}")
                self.model = YOLO(model_path, task="detect")
                self.model_type = "TensorRT"
            else:
                logger.info(f"Loading PyTorch model from {model_path}")
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.model_type = "PyTorch"

            # Handle class names for both model types
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            else:
                # For TensorRT models, try to load class names from corresponding .pt file
                if is_tensorrt:
                    pt_path = os.path.splitext(model_path)[0] + ".pt"
                    if os.path.exists(pt_path):
                        logger.info(f"Loading class names from {pt_path}")
                        temp_model = YOLO(pt_path)
                        self.class_names = temp_model.names
                        del temp_model
                    else:
                        logger.warning(f"Could not find {pt_path} to load class names. Class names will be empty.")
                        self.class_names = {}
                else:
                    self.class_names = {}

            logger.info(f"{self.model_type} model loaded successfully from {model_path}. Classes: {len(self.class_names)}")

        except Exception as e:
            if is_tensorrt:
                logger.error(f"Failed to load TensorRT model: {e}")
                logger.info("Falling back to regular PyTorch model...")
                self._load_fallback_model(model_path)
            else:
                raise

    def _export_to_tensorrt(self, engine_path: str) -> None:
        """Export PyTorch model to TensorRT format."""
        pt_path = os.path.splitext(engine_path)[0] + ".pt"
        try:
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"PyTorch model not found at {pt_path} for TensorRT export.")

            logger.info(f"Exporting {pt_path} to TensorRT format...")
            
            model = YOLO(pt_path)
            
            model.export(
                format="engine",
                half=True,
            )
            
            logger.info(f"TensorRT export completed successfully for {engine_path}")
            
        except Exception as e:
            logger.error(f"Failed to export to TensorRT from {pt_path}: {e}")
            raise

    def _load_fallback_model(self, engine_path: str) -> None:
        """Load regular PyTorch model as fallback."""
        pt_path = os.path.splitext(engine_path)[0] + ".pt"
        try:
            if not os.path.exists(pt_path):
                 raise FileNotFoundError(f"PyTorch fallback model not found at {pt_path}.")

            logger.info(f"Loading fallback PyTorch model from {pt_path}")
            self._load_model_with_fallback(pt_path, is_tensorrt=False)
        except Exception as e:
            logger.error(f"Failed to load fallback model from {pt_path}: {e}")
            raise
