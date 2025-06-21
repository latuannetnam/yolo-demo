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
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup the computation device (GPU/CPU)."""
        if Config.validate_gpu_availability():
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
class SegmentationDetector:
    """YOLOv11-based instance segmentation detector."""
    
    def __init__(self):
        """Initialize the segmentation detector."""
        self.model = None
        self.device = self._setup_device()
        self.class_names: Dict[int, str] = {}
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup the computation device (GPU/CPU)."""
        if Config.validate_gpu_availability():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def _load_model(self) -> None:
        """Load the YOLOv11 segmentation model."""
        try:
            MODEL_PATH = Config.get_full_segmentation_model_name()
            self.model = YOLO(MODEL_PATH)
            self.model.to(self.device)
            
            # Get class names
            self.class_names = self.model.names
            logger.info(f"Segmentation model loaded successfully from {MODEL_PATH}. Classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise
    
    def detect_and_segment(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform instance segmentation on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class info, and masks
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(
                frame,
                conf=Config.MODEL_CONFIDENCE,
                iou=Config.MODEL_IOU_THRESHOLD,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and result.masks is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    masks = result.masks.data.cpu().numpy()
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Get the mask for this detection
                        mask = masks[i] if i < len(masks) else None
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self.class_names.get(class_id, f'Class_{class_id}'),
                            'mask': mask
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Segmentation detection error: {e}")
            return []
    
    def draw_segmentations(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                          camera_id: int = 0, alpha: float = 0.5) -> np.ndarray:
        """
        Draw segmentation masks and bounding boxes on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            camera_id: Camera identifier for color selection
            alpha: Transparency for mask overlay
            
        Returns:
            Frame with drawn segmentations
        """
        if not detections:
            return frame
        
        # Create overlay for masks
        overlay = frame.copy()
        
        # Select color based on camera ID
        bbox_color = Config.BBOX_COLORS[camera_id % len(Config.BBOX_COLORS)]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            mask = detection.get('mask')
            
            # Draw segmentation mask if available
            if mask is not None:
                # Resize mask to frame size if needed
                if mask.shape != frame.shape[:2]:
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                else:
                    mask_resized = mask
                
                # Convert mask to binary
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Create colored mask
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :] = bbox_color
                
                # Apply mask
                overlay[mask_binary > 0] = cv2.addWeighted(
                    overlay[mask_binary > 0], 
                    1 - alpha, 
                    mask_colored[mask_binary > 0], 
                    alpha, 
                    0
                )
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            
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
                bbox_color,
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
        
        # Blend the overlay with the original frame
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result


class PromptDetector:
    """YOLOE-based prompt object detector."""
    
    def __init__(self):
        """Initialize the prompt detector."""
        self.model = None
        self.device = self._setup_device()
        self.class_names: Dict[int, str] = {}
        self.current_prompts: List[str] = []
        self._load_model()
    
    def _setup_device(self) -> str:
        """Setup the computation device (GPU/CPU)."""
        if Config.validate_gpu_availability():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def _load_model(self) -> None:
        """Load the YOLOE model."""
        try:
            # Try to import YOLOE specifically, fall back to YOLO
            try:
                from ultralytics import YOLOE
                MODEL_PATH = Config.get_full_prompt_model_name()
                self.model = YOLOE(MODEL_PATH)
            except ImportError:
                logger.warning("YOLOE not available, using regular YOLO model")
                from ultralytics import YOLO
                MODEL_PATH = Config.get_full_prompt_model_name()
                self.model = YOLO(MODEL_PATH)
            
            self.model.to(self.device)
            
            # Get default class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                self.class_names = {}
            logger.info(f"YOLOE model loaded successfully from {MODEL_PATH}. Default classes: {len(self.class_names)}")
        
        except Exception as e:
            logger.error(f"Failed to load YOLOE model: {e}")
            logger.warning("Make sure you have the YOLOE model installed. You can download it from Ultralytics.")
            raise
    
    def set_prompts(self, prompts: List[str]) -> bool:
        """
        Set text prompts for detection.
        
        Args:
            prompts: List of text prompts (class names) to detect
            
        Returns:
            True if prompts were set successfully, False otherwise
        """
        if self.model is None:
            logger.error("Model not loaded")
            return False
        
        try:
            # Set prompts for YOLOE model
            if hasattr(self.model, 'set_classes') and hasattr(self.model, 'get_text_pe'):
                # This is a YOLOE model
                self.model.set_classes(prompts, self.model.get_text_pe(prompts))
            else:
                # Fallback for regular YOLO models - just store prompts
                logger.warning("Model doesn't support prompt setting. Storing prompts for reference only.")
            
            self.current_prompts = prompts.copy()
            
            # Update class names mapping
            self.class_names = {i: name for i, name in enumerate(prompts)}
            
            logger.info(f"Prompts set successfully: {prompts}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set prompts: {e}")
            return False
    
    def get_current_prompts(self) -> List[str]:
        """Get currently set prompts."""
        return self.current_prompts.copy()
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform prompt-based object detection on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class info
        """
        if self.model is None:
            return []
        
        if not self.current_prompts:
            logger.warning("No prompts set. Use set_prompts() to define what to detect.")
            return []
        
        try:
            # Run inference
            results = self.model(
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
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Get class name from current prompts
                        class_name = self.current_prompts[class_id] if class_id < len(self.current_prompts) else f'Class_{class_id}'
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'prompt': class_name
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Prompt detection error: {e}")
            return []
    
    def detect_and_segment(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform prompt-based object detection and segmentation on a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class info, and masks
        """
        if self.model is None:
            return []
        
        if not self.current_prompts:
            logger.warning("No prompts set. Use set_prompts() to define what to detect.")
            return []
        
        try:
            # Run inference
            results = self.model(
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
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Get masks if available
                    masks = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Get class name from current prompts
                        class_name = self.current_prompts[class_id] if class_id < len(self.current_prompts) else f'Class_{class_id}'
                        
                        # Get the mask for this detection
                        mask = masks[i] if masks is not None and i < len(masks) else None
                        
                        detection = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': class_name,
                            'prompt': class_name,
                            'mask': mask
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Prompt detection and segmentation error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                       camera_id: int = 0) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame for prompt-based detections.
        
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
            prompt = detection.get('prompt', class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text with prompt indicator
            label = f"[{prompt}]: {confidence:.2f}"
            
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
    
    def draw_segmentations(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                          camera_id: int = 0, alpha: float = 0.5) -> np.ndarray:
        """
        Draw segmentation masks and bounding boxes on the frame for prompt-based detections.
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            camera_id: Camera identifier for color selection
            alpha: Transparency for mask overlay
            
        Returns:
            Frame with drawn segmentations
        """
        if not detections:
            return frame
        
        # Create overlay for masks
        overlay = frame.copy()
        
        # Select color based on camera ID
        bbox_color = Config.BBOX_COLORS[camera_id % len(Config.BBOX_COLORS)]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            prompt = detection.get('prompt', class_name)
            mask = detection.get('mask')
            
            # Draw segmentation mask if available
            if mask is not None:
                # Resize mask to frame size if needed
                if mask.shape != frame.shape[:2]:
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                else:
                    mask_resized = mask
                
                # Convert mask to binary
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Create colored mask
                mask_colored = np.zeros_like(frame)
                mask_colored[:, :] = bbox_color
                
                # Apply mask
                overlay[mask_binary > 0] = cv2.addWeighted(
                    overlay[mask_binary > 0], 
                    1 - alpha, 
                    mask_colored[mask_binary > 0], 
                    alpha, 
                    0
                )
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Prepare label text with prompt indicator
            label = f"[{prompt}]: {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - baseline - 10),
                (x1 + label_width, y1),
                bbox_color,
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
        
        # Blend the overlay with the original frame
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result
