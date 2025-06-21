"""
Real-time prompt-based object detection system with TensorRT optimization and multi-camera support.
"""
import cv2
import time
import signal
import sys
import os
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger
from ultralytics import YOLOE
from ultralytics import YOLO
import torch

from config import Config
from camera_stream import MultiCameraManager

# Setup logging
# logger.add("logs/tensorrt_prompt_app.log", rotation="10 MB", retention="7 days", level="INFO")

class TensorRTPromptDetector:
    """TensorRT-optimized YOLOE prompt object detector."""
    
    def __init__(self):
        """Initialize the TensorRT prompt detector."""
        self.model = None
        self.device = self._setup_device()
        self.class_names: Dict[int, str] = {}
        self.current_prompts: List[str] = []
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
        model_name = Config.TENSORRT_PROMPT_MODEL_NAME
        return os.path.join(Config.MODEL_DIR, model_name)
    
    def _load_tensorrt_model(self) -> None:
        """Load the TensorRT optimized YOLOE model."""
        try:
            # Check if TensorRT engine exists
            engine_path = self._get_tensorrt_model_path()
            
            if not os.path.exists(engine_path):
                logger.warning(f"TensorRT engine not found at {engine_path}")
                logger.info("Attempting to export PyTorch model to TensorRT...")
                self._export_to_tensorrt()
            
            # Load the TensorRT engine
            logger.info(f"Loading TensorRT YOLOE model from {engine_path}")
            self.model = YOLOE(engine_path, verbose=True)            
            
            # # Get class names (from original model if needed)
            # if hasattr(self.model, 'names') and self.model.names:
            #     self.class_names = self.model.names
            # else:
            #     # Fallback: load from original model to get class names
            #     original_model_path = Config.get_full_tensorrt_prompt_model_name()
            #     try:
            #         from ultralytics import YOLOE
            #         temp_model = YOLOE(original_model_path)
            #     except ImportError:
            #         temp_model = YOLO(original_model_path)
            #     self.class_names = temp_model.names if hasattr(temp_model, 'names') else {}
            #     del temp_model
            
            logger.info(f"TensorRT YOLOE model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT YOLOE model: {e}")
            logger.info("Falling back to regular PyTorch YOLOE model...")
            self._load_fallback_model()
    
    def _export_to_tensorrt(self) -> None:
        """Export PyTorch YOLOE model to TensorRT format."""
        try:
            full_model_path = Config.get_full_tensorrt_prompt_model_name()
            # Extract model name without extension
            model_name = os.path.splitext(os.path.basename(full_model_path))[0]
            original_model_path = os.path.join(Config.MODEL_DIR, model_name + ".pt")
            
            logger.info(f"Exporting {original_model_path} to TensorRT format...")
            
            # Try to load YOLOE specifically
            try:
                from ultralytics import YOLOE
                model = YOLOE(original_model_path)
            except ImportError:
                logger.warning("YOLOE not available, using regular YOLO model")
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
            
            logger.info("TensorRT YOLOE export completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to export YOLOE to TensorRT: {e}")
            raise
    
    def _load_fallback_model(self) -> None:
        """Load regular PyTorch YOLOE model as fallback."""
        try:
            full_model_path = Config.get_full_tensorrt_prompt_model_name()
            # Extract model name without extension
            model_name = os.path.splitext(os.path.basename(full_model_path))[0]
            original_model_path = os.path.join(Config.MODEL_DIR, model_name + ".pt")
            
            # Try to load YOLOE specifically
            self.model = YOLOE(original_model_path)                
            self.model.to(self.device)
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            logger.info(f"Fallback PyTorch YOLOE model loaded from {original_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load fallback YOLOE model: {e}")
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
            if (self.model is not None and 
                hasattr(self.model, 'set_classes') and 
                hasattr(self.model, 'get_text_pe')):
                # This is a YOLOE model with prompt support
                # Type ignore for dynamic method calls
                text_pe = self.model.get_text_pe(prompts)  # type: ignore
                self.model.set_classes(prompts, text_pe)  # type: ignore
            else:
                # Fallback for regular YOLO models - just store prompts
                logger.warning("Model doesn't support prompt setting. Storing prompts for reference only.")
            
            self.current_prompts = prompts.copy()
            
            # Update class names mapping
            self.class_names = {i: name for i, name in enumerate(prompts)}
            
            logger.info(f"TensorRT prompts set successfully: {prompts}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set TensorRT prompts: {e}")
            return False
    
    def get_current_prompts(self) -> List[str]:
        """Get currently set prompts."""
        return self.current_prompts.copy()
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform prompt-based object detection on a frame using TensorRT optimized model.
        
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
            logger.error(f"TensorRT prompt detection error: {e}")
            return []
    
    def detect_and_segment(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform prompt-based object detection and segmentation on a frame using TensorRT.
        
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
                    
                    # Handle masks if available (for segmentation models)
                    masks = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data
                        if isinstance(masks, torch.Tensor):
                            masks = masks.cpu().numpy()
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().numpy()
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
                        
                        # Add mask if available
                        if masks is not None and i < len(masks):
                            detection['mask'] = masks[i]
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"TensorRT prompt detection and segmentation error: {e}")
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
            prompt = detection.get('prompt', class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text with prompt info
            label = f"{prompt}: {confidence:.2f}"
            
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
                          camera_id: int = 0) -> np.ndarray:
        """
        Draw bounding boxes, labels, and segmentation masks on the frame.
        
        Args:
            frame: Input image frame
            detections: List of detection dictionaries with optional masks
            camera_id: Camera identifier for color selection
            
        Returns:
            Frame with drawn detections and segmentations
        """
        if not detections:
            return frame
        
        # Select color based on camera ID
        color = Config.BBOX_COLORS[camera_id % len(Config.BBOX_COLORS)]
        
        # Create overlay for masks
        overlay = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            prompt = detection.get('prompt', class_name)
            
            # Draw segmentation mask if available
            if 'mask' in detection and detection['mask'] is not None:
                mask = detection['mask']
                
                # Resize mask to frame dimensions if needed
                if mask.shape != (frame.shape[0], frame.shape[1]):
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                
                # Create colored mask
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0.5] = color
                
                # Blend with overlay
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.3, 0)
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text with prompt info
            label = f"{prompt}: {confidence:.2f}"
            
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                overlay,
                (x1, y1 - label_height - baseline - 10),
                (x1 + label_width, y1),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                overlay,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return overlay


class TensorRTPromptDetectionSystem:
    """Main TensorRT prompt-based object detection system."""
    
    def __init__(self, initial_prompts: Optional[List[str]] = None):
        """Initialize the TensorRT prompt detection system."""
        self.detector: Optional[TensorRTPromptDetector] = None
        self.camera_manager = MultiCameraManager()
        self.running = False
        self.stats: Dict[str, Any] = {
            'total_frames': 0,
            'detection_time': 0.0,
            'display_time': 0.0
        }
        self.initial_prompts = initial_prompts if initial_prompts else ["car"]  # Default prompts
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize the TensorRT prompt detection system components."""
        try:
            # Initialize TensorRT prompt detector
            logger.info("Initializing TensorRT prompt-based object detector...")
            self.detector = TensorRTPromptDetector()
            
            # Set initial prompts
            if self.initial_prompts:
                logger.info(f"Setting initial prompts: {self.initial_prompts}")
                if not self.detector.set_prompts(self.initial_prompts):
                    logger.error("Failed to set initial prompts. Please check model and prompts.")
                    # Allow to continue without prompts, user can set them later
            
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
            
            logger.info(f"TensorRT prompt system initialized with {len(active_cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT prompt system: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> np.ndarray:
        """
        Process a single frame with TensorRT prompt-based object detection.
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            Processed frame with detections
        """
        try:
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            # Ensure detector is available
            if self.detector is None:
                logger.error("Detector not initialized")
                return frame
            
            # Perform TensorRT prompt-based object detection and segmentation
            start_time = time.time()
            # Using detect_and_segment for YOLOE models as they usually support it
            detections = self.detector.detect_and_segment(frame)
            detection_time = time.time() - start_time
            
            # Draw detections and segmentations
            frame_with_detections = self.detector.draw_segmentations(frame, detections, camera_id)
            
            # Add FPS and camera info
            if Config.DISPLAY_FPS:
                fps = self.camera_manager.get_camera_fps(camera_id)
                current_prompts_str = ", ".join(self.detector.get_current_prompts()) if self.detector else "None"
                info_text = f"Cam {camera_id} (TensorRT) | FPS: {fps:.1f} | Objects: {len(detections)}"
                prompt_text = f"Prompts: {current_prompts_str if current_prompts_str else 'None'}"
                
                cv2.putText(
                    frame_with_detections,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame_with_detections,
                    prompt_text,
                    (10, 60),  # Position below FPS info
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),  # Cyan color for prompt text
                    1
                )
                
                # Add detection time info
                time_text = f"Detection: {detection_time*1000:.1f}ms"
                cv2.putText(
                    frame_with_detections,
                    time_text,
                    (10, 90),  # Position below prompt text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['detection_time'] = float(self.stats['detection_time']) + detection_time
            
            return frame_with_detections
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return frame
    
    def run(self) -> None:
        """Run the main TensorRT prompt detection loop."""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting TensorRT prompt-based object detection system...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - TensorRT Prompt Detection"
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
                    window_name = f"Camera {camera_id} - TensorRT Prompt Detection"
                    cv2.imshow(window_name, processed_frame)
                    
                    display_time = time.time() - start_time
                    self.stats['display_time'] = float(self.stats['display_time']) + display_time
                
                # Check for exit key or prompt change
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # 's' key to save screenshot
                    self._save_screenshots(frames)
                elif key == ord('r'):  # 'r' key to reset stats
                    self._reset_stats()
                elif key == ord('p'):  # 'p' key to change prompts
                    self._change_prompts()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def _change_prompts(self):
        """Allow user to change prompts at runtime."""
        try:
            if self.detector is None:
                logger.error("Detector not initialized")
                return
                
            current_prompts_str = ", ".join(self.detector.get_current_prompts())
            new_prompts_str = input(f"Enter new prompts (comma-separated, current: [{current_prompts_str}]): ")
            if new_prompts_str:
                new_prompts = [p.strip() for p in new_prompts_str.split(',') if p.strip()]
                if new_prompts:
                    logger.info(f"Changing TensorRT prompts to: {new_prompts}")
                    if self.detector.set_prompts(new_prompts):
                        logger.info("TensorRT prompts updated successfully.")
                    else:
                        logger.error("Failed to update TensorRT prompts.")
                else:
                    logger.info("No valid prompts entered.")
            else:
                logger.info("Prompt change cancelled or no input.")
        except Exception as e:
            logger.error(f"Error changing TensorRT prompts: {e}")
    
    def _save_screenshots(self, frames: Dict[int, np.ndarray]) -> None:
        """Save screenshots of current frames."""
        timestamp = int(time.time())
        for camera_id, frame in frames.items():
            # Re-process frame to include latest detections and info
            processed_frame = self.process_frame(frame, camera_id)
            filename = f"tensorrt_prompt_screenshot_cam_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"TensorRT prompt screenshot saved: {filename}")
    
    def _reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_frames': 0,
            'detection_time': 0.0,
            'display_time': 0.0
        }
        logger.info("TensorRT prompt statistics reset")
    
    def stop(self) -> None:
        """Stop the TensorRT prompt detection system."""
        self.running = False
        
        # Stop all cameras
        self.camera_manager.stop_all()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.stats['total_frames'] > 0:
            avg_detection_time = self.stats['detection_time'] / self.stats['total_frames']
            avg_display_time = self.stats['display_time'] / self.stats['total_frames']
            
            logger.info("=== Final TensorRT Prompt Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average detection time: {avg_detection_time*1000:.2f}ms")
            logger.info(f"Average display time: {avg_display_time*1000:.2f}ms")
        
        logger.info("TensorRT prompt-based object detection system stopped")


def main():
    """Main function."""
    print("🚀 Real-time Prompt-Based Object Detection System with TensorRT (YOLOE) v1.0")
    print("==============================================================================")
    print("Features:")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time YOLOE object detection with text prompts")
    print("- TensorRT optimization for maximum performance")
    print("- GPU acceleration with TensorRT engine")
    print("- Automatic PyTorch to TensorRT conversion")
    print("- Separate display windows for each camera")
    print("- Dynamic prompt changing")
    print("- Segmentation support (if model supports it)")
    print("- Initial prompts configured via INITIAL_PROMPTS environment variable")
    print("- TensorRT model configured via TENSORRT_PROMPT_MODEL_NAME environment variable")
    print()
    print("Controls:")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save screenshots")
    print("- Press 'r' to reset statistics")
    print("- Press 'p' to change detection prompts")
    print()
    
    # Get initial prompts from environment variable
    initial_prompts_list = Config.get_initial_prompts()
    logger.info(f"Using initial prompts from environment: {initial_prompts_list}")
    logger.info(f"Using TensorRT prompt model: {Config.TENSORRT_PROMPT_MODEL_NAME}")
    
    # Create and run the TensorRT prompt detection system
    system = TensorRTPromptDetectionSystem(initial_prompts=initial_prompts_list)
    
    try:
        system.run()
    except Exception as e:
        logger.error(f"TensorRT prompt system error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
