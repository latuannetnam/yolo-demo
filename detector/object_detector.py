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
import supervision as sv
from supervision.annotators.utils import ColorLookup

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

        self.tracker = sv.ByteTrack()
        self.smoother = sv.DetectionsSmoother()
        self.box_annotator = sv.BoxAnnotator(color_lookup=ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK, color_lookup=ColorLookup.INDEX
        )
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.CENTER,
            trace_length=30,
            thickness=2,
            color_lookup=ColorLookup.INDEX,
        )
    
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
    
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class info
        """
        if self.model is None:
            return sv.Detections.empty()
        
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=Config.MODEL_CONFIDENCE,
                iou=Config.MODEL_IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            detections = sv.Detections.from_ultralytics(results[0])
            detections = self.tracker.update_with_detections(detections)
            detections = self.smoother.update_with_detections(detections)
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return sv.Detections.empty()
    
    def draw_detections(self, frame: np.ndarray, detections: sv.Detections, 
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
        if len(detections) == 0:
            return frame

        labels = []
        if detections.class_id is not None and detections.confidence is not None:
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]
                
                label_parts = []

                if detections.tracker_id is not None:
                    tracker_id = detections.tracker_id[i]
                    label_parts.append(f"#{tracker_id}")
                
                label_parts.append(self.class_names.get(class_id, f"ID {class_id}"))
                label_parts.append(f"{confidence:.2f}")

                labels.append(" ".join(label_parts))

        annotated_frame = self.trace_annotator.annotate(frame.copy(), detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        if labels:
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels
            )

        return annotated_frame


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
