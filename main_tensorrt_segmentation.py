"""
Real-time instance segmentation system with TensorRT optimization and multi-camera support.
"""
import cv2
import time
import signal
import sys
import os
from typing import Dict, Any, List
import numpy as np
from loguru import logger
from ultralytics import YOLO
import torch

from config import Config
from camera_stream import MultiCameraManager

# Setup logging
# logger.add("logs/tensorrt_segmentation_app.log", rotation="10 MB", retention="7 days", level="INFO")

class TensorRTSegmentationDetector:
    """TensorRT-optimized YOLO instance segmentation detector."""
    
    def __init__(self):
        """Initialize the TensorRT segmentation detector."""
        self.model = None
        self.device = self._setup_device()
        self.class_names: Dict[int, str] = {}
        self._load_tensorrt_model()
    
    def _setup_device(self) -> str:
        """Setup the computation device (GPU/CPU)."""
        if Config.validate_gpu_availability():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def _get_tensorrt_model_path(self) -> str:
        """Get the TensorRT engine model path."""
        # Use the TensorRT segmentation model name from config
        model_name = Config.TENSORRT_SEGMENTATION_MODEL_NAME
        
        return os.path.join(Config.MODEL_DIR, model_name)
    
    def _load_tensorrt_model(self) -> None:
        """Load the TensorRT optimized YOLO segmentation model."""
        try:
            import os
            
            # Check if TensorRT engine exists
            engine_path = self._get_tensorrt_model_path()
            
            if not os.path.exists(engine_path):
                logger.warning(f"TensorRT engine not found at {engine_path}")
                logger.info("Attempting to export PyTorch segmentation model to TensorRT...")
                self._export_to_tensorrt()
            
            # Load the TensorRT engine
            logger.info(f"Loading TensorRT segmentation model from {engine_path}")
            self.model = YOLO(engine_path, task="segment")
            
            # Get class names (from original model if needed)
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            else:
                # Fallback: load from original model to get class names
                original_model_path = Config.get_full_tensorrt_segmentation_model_name()
                temp_model = YOLO(original_model_path)
                self.class_names = temp_model.names
                del temp_model
            
            logger.info(f"TensorRT segmentation model loaded successfully. Classes: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT segmentation model: {e}")
            logger.info("Falling back to regular PyTorch segmentation model...")
            self._load_fallback_model()
    
    def _export_to_tensorrt(self) -> None:
        """Export PyTorch segmentation model to TensorRT format."""
        try:
            full_model_path = Config.get_full_tensorrt_segmentation_model_name()
            # Extract model name without extension and replace with .pt
            model_name = os.path.splitext(os.path.basename(full_model_path))[0]
            original_model_path = os.path.join(Config.MODEL_DIR, model_name + ".pt")
            
            logger.info(f"Exporting {original_model_path} to TensorRT format...")
            
            model = YOLO(original_model_path)
            
            # Export to TensorRT engine format
            model.export(
                format="engine",
                # You can uncomment and adjust these parameters for optimization:
                # dynamic=True,  # Enable dynamic input shapes
                # batch=8,       # Batch size for optimization
                # workspace=4,   # Max workspace size in GB
                # int8=True,     # Enable INT8 quantization (requires calibration data)
                # data="coco.yaml",  # Dataset for INT8 calibration
            )
            
            logger.info("TensorRT segmentation export completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to export segmentation model to TensorRT: {e}")
            raise
    
    def _load_fallback_model(self) -> None:
        """Load regular PyTorch segmentation model as fallback."""
        try:
            full_model_path = Config.get_full_tensorrt_segmentation_model_name()
            # Extract model name without extension and replace with .pt
            model_name = os.path.splitext(os.path.basename(full_model_path))[0]
            original_model_path = os.path.join(Config.MODEL_DIR, model_name + ".pt")
            
            self.model = YOLO(original_model_path)
            self.model.to(self.device)
            self.class_names = self.model.names
            logger.info(f"Fallback PyTorch segmentation model loaded from {original_model_path}")
        except Exception as e:
            logger.error(f"Failed to load fallback segmentation model: {e}")
            raise
    
    def detect_and_segment(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform instance segmentation on a frame using TensorRT optimized model.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries containing bbox, confidence, class info, and masks
        """
        if self.model is None:
            return []
        
        try:
            # Run inference with TensorRT model
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
                
                if result.boxes is not None and result.masks is not None:
                    # Handle both tensor and numpy array cases
                    boxes = result.boxes.xyxy
                    confidences = result.boxes.conf
                    class_ids = result.boxes.cls
                    masks = result.masks.data
                    
                    # Convert to numpy arrays safely
                    import torch
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().numpy()
                    if isinstance(confidences, torch.Tensor):
                        confidences = confidences.cpu().numpy()
                    if isinstance(class_ids, torch.Tensor):
                        class_ids = class_ids.cpu().numpy()
                    if isinstance(masks, torch.Tensor):
                        masks = masks.cpu().numpy()
                    
                    # Ensure class_ids are integers
                    class_ids = class_ids.astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().numpy()
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
            logger.error(f"TensorRT segmentation detection error: {e}")
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


class TensorRTInstanceSegmentationSystem:
    """Main TensorRT instance segmentation system."""
    
    def __init__(self):
        """Initialize the TensorRT segmentation system."""
        self.detector: TensorRTSegmentationDetector | None = None
        self.camera_manager = MultiCameraManager()
        self.running = False
        self.stats: Dict[str, Any] = {
            'total_frames': 0,
            'segmentation_time': 0.0,
            'display_time': 0.0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize the TensorRT segmentation system components."""
        try:
            # Initialize TensorRT segmentation detector
            logger.info("Initializing TensorRT instance segmentation detector...")
            self.detector = TensorRTSegmentationDetector()
            
            # Setup cameras
            logger.info("Setting up cameras...")
            streams = Config.get_stream_config()
            
            for i, stream in enumerate(streams):
                if self.camera_manager.add_camera(stream, i):
                    logger.info(f"Camera {i} added successfully")
                else:
                    logger.error(f"Failed to add camera {i}: {stream}")
            
            active_cameras = self.camera_manager.get_active_cameras()
            if not active_cameras:
                logger.error("No cameras available. Exiting.")
                return False
            
            logger.info(f"TensorRT segmentation system initialized with {len(active_cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT segmentation system: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> np.ndarray:
        """
        Process a single frame with TensorRT instance segmentation.
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            Processed frame with segmentations
        """
        try:
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            # Check if detector is initialized
            if self.detector is None:
                logger.error("Detector not initialized")
                return frame
            
            # Perform TensorRT instance segmentation
            start_time = time.time()
            detections = self.detector.detect_and_segment(frame)
            segmentation_time = time.time() - start_time
            
            # Draw segmentations
            frame_with_segmentations = self.detector.draw_segmentations(frame, detections, camera_id)
            
            # Add FPS and camera info
            if Config.DISPLAY_FPS:
                fps = self.camera_manager.get_camera_fps(camera_id)
                info_text = f"Camera {camera_id} (TensorRT) | FPS: {fps:.1f} | Objects: {len(detections)}"
                cv2.putText(
                    frame_with_segmentations,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Add segmentation time info
                time_text = f"Segmentation: {segmentation_time*1000:.1f}ms"
                cv2.putText(
                    frame_with_segmentations,
                    time_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['segmentation_time'] = float(self.stats['segmentation_time']) + segmentation_time
            
            return frame_with_segmentations
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return frame
    
    def run(self) -> None:
        """Run the main TensorRT segmentation loop."""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting TensorRT instance segmentation system...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - TensorRT Instance Segmentation"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
            
            # Position windows
            x, y = self.camera_manager.get_window_position(camera_id)
            cv2.moveWindow(window_name, x, y)
        
        try:
            while self.running:
                # Get frames from all cameras
                frames = self.camera_manager.get_all_frames()
                
                if not frames:
                    time.sleep(0.01)
                    continue
                
                # Process each frame
                for camera_id, frame in frames.items():
                    start_time = time.time()
                    
                    processed_frame = self.process_frame(frame, camera_id)
                    
                    # Display frame
                    window_name = f"Camera {camera_id} - TensorRT Instance Segmentation"
                    cv2.imshow(window_name, processed_frame)
                    
                    display_time = time.time() - start_time
                    self.stats['display_time'] = float(self.stats['display_time']) + display_time
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # 's' key to save screenshot
                    self._save_screenshots(frames)
                elif key == ord('r'):  # 'r' key to reset stats
                    self._reset_stats()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def _save_screenshots(self, frames: Dict[int, np.ndarray]) -> None:
        """Save screenshots of current frames."""
        timestamp = int(time.time())
        for camera_id, frame in frames.items():
            processed_frame = self.process_frame(frame, camera_id)
            filename = f"tensorrt_segmentation_screenshot_camera_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"TensorRT segmentation screenshot saved: {filename}")
    
    def _reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_frames': 0,
            'segmentation_time': 0.0,
            'display_time': 0.0
        }
        logger.info("Statistics reset")
    
    def stop(self) -> None:
        """Stop the TensorRT segmentation system."""
        self.running = False
        
        # Stop all cameras
        self.camera_manager.stop_all()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.stats['total_frames'] > 0:
            avg_segmentation_time = self.stats['segmentation_time'] / self.stats['total_frames']
            avg_display_time = self.stats['display_time'] / self.stats['total_frames']
            
            logger.info("=== Final TensorRT Segmentation Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average segmentation time: {avg_segmentation_time*1000:.2f}ms")
            logger.info(f"Average display time: {avg_display_time*1000:.2f}ms")
        
        logger.info("TensorRT instance segmentation system stopped")


def main():
    """Main function."""
    print("🎯 Real-time Instance Segmentation System with TensorRT v1.0")
    print("==============================================================")
    print("Features:")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time YOLOv11 instance segmentation with TensorRT optimization")
    print("- GPU acceleration with TensorRT engine")
    print("- Automatic PyTorch to TensorRT conversion")
    print("- Segmentation masks with transparency overlay")
    print("- Separate display windows for each camera")
    print()
    print("Controls:")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save screenshots")
    print("- Press 'r' to reset statistics")
    print()
    
    # Create and run the TensorRT segmentation system
    system = TensorRTInstanceSegmentationSystem()
    
    try:
        system.run()
    except Exception as e:
        logger.error(f"TensorRT segmentation system error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
