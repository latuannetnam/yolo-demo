"""
Alternative heatmap implementation using custom heatmap overlay on YOLO detections.
This provides a fallback if the Ultralytics solutions.Heatmap doesn't work as expected.
"""
import cv2
import time
import signal
import sys
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from loguru import logger
from ultralytics import YOLO

from config import Config
from camera_stream import MultiCameraManager

class HeatmapSystem:
    """Object detection system with custom heatmap visualization."""
    
    def __init__(self):
        """Initialize the custom heatmap system."""
        self.model = None
        self.camera_manager = MultiCameraManager()
        self.running = False
        self.heatmaps = {}  # Dictionary to store accumulated heatmaps for each camera
        self.stats = {
            'total_frames': 0,
            'detection_time': 0.0,
            'display_time': 0.0
        }
        
        # Heatmap configuration
        self.heatmap_config = {
            'colormap': cv2.COLORMAP_JET,
            'alpha': 0.4,  # Transparency of heatmap overlay
            'decay_factor': 0.99,  # How fast the heatmap fades
            'min_confidence': Config.MODEL_CONFIDENCE,
            'blur_kernel': 15,  # Gaussian blur kernel size for smoother heatmap
            'point_size': 30,   # Size of the point to add to heatmap
            'show_boxes': True  # Whether to draw bounding boxes on detections
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()
    
    def _load_model(self) -> bool:
        """Load YOLO model."""
        try:
            model_path = Config.get_full_model_name()
            self.model = YOLO(model_path)
            
            # Set device
            device = "cuda" if Config.validate_gpu_availability() else "cpu"
            self.model.to(device)
            
            logger.info(f"Model loaded successfully: {model_path} on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _initialize_heatmap(self, camera_id: int, frame_shape: Tuple[int, int]) -> None:
        """Initialize heatmap for a camera."""
        height, width = frame_shape[:2]
        self.heatmaps[camera_id] = np.zeros((height, width), dtype=np.float32)
        logger.info(f"Heatmap initialized for camera {camera_id}: {width}x{height}")
    
    def _update_heatmap(self, camera_id: int, detections: List[Dict], frame_shape: Tuple[int, int]) -> None:
        """Update heatmap with new detections."""
        if camera_id not in self.heatmaps:
            self._initialize_heatmap(camera_id, frame_shape)
        
        # Apply decay to existing heatmap
        self.heatmaps[camera_id] *= self.heatmap_config['decay_factor']
        
        # Add new detections to heatmap
        for detection in detections:
            if detection['confidence'] >= self.heatmap_config['min_confidence']:
                # Get center point of bounding box
                bbox = detection['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Add Gaussian blob to heatmap
                self._add_gaussian_blob(camera_id, center_x, center_y, detection['confidence'])
    
    def _add_gaussian_blob(self, camera_id: int, x: int, y: int, intensity: float) -> None:
        """Add a Gaussian blob to the heatmap at the specified location."""
        heatmap = self.heatmaps[camera_id]
        height, width = heatmap.shape
        
        # Create Gaussian kernel
        size = self.heatmap_config['point_size']
        kernel = np.zeros((size * 2 + 1, size * 2 + 1), dtype=np.float32)
        
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                dx = i - size
                dy = j - size
                distance = np.sqrt(dx * dx + dy * dy)
                kernel[i, j] = intensity * np.exp(-(distance * distance) / (2 * (size / 3) ** 2))
        
        # Add kernel to heatmap at the specified location
        start_x = max(0, x - size)
        end_x = min(width, x + size + 1)
        start_y = max(0, y - size)
        end_y = min(height, y + size + 1)
        
        kernel_start_x = max(0, size - x)
        kernel_end_x = kernel_start_x + (end_x - start_x)
        kernel_start_y = max(0, size - y)
        kernel_end_y = kernel_start_y + (end_y - start_y)
        
        heatmap[start_y:end_y, start_x:end_x] += kernel[kernel_start_y:kernel_end_y, kernel_start_x:kernel_end_x]
    
    def _create_heatmap_overlay(self, camera_id: int, frame: np.ndarray) -> np.ndarray:
        """Create heatmap overlay on the frame."""
        if camera_id not in self.heatmaps:
            return frame
        
        heatmap = self.heatmaps[camera_id]
        
        # Normalize heatmap to 0-255 range
        if heatmap.max() > 0:
            normalized_heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            normalized_heatmap = heatmap.astype(np.uint8)
        
        # Apply Gaussian blur for smoother visualization
        if self.heatmap_config['blur_kernel'] > 0:
            normalized_heatmap = cv2.GaussianBlur(
                normalized_heatmap, 
                (self.heatmap_config['blur_kernel'], self.heatmap_config['blur_kernel']), 
                0
            )
        
        # Apply colormap
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.heatmap_config['colormap'])
        
        # Blend with original frame
        alpha = self.heatmap_config['alpha']
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        return overlay
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame and return detection results."""
        try:
            results = self.model(frame, conf=self.heatmap_config['min_confidence'])
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        })
            
            return detections
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection bounding boxes and labels on frame."""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - label_size[1] - 10)), 
                         (int(bbox[0] + label_size[0]), int(bbox[1])), (0, 255, 0), -1)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1] - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> np.ndarray:
        """Process frame with object detection and heatmap overlay."""
        try:
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            # Detect objects
            start_time = time.time()
            detections = self._detect_objects(frame)
            detection_time = time.time() - start_time
            
            # Update heatmap with detections
            self._update_heatmap(camera_id, detections, frame.shape)
            
            # Create heatmap overlay
            heatmap_frame = self._create_heatmap_overlay(camera_id, frame)
            
            # Conditionally draw detection boxes based on configuration
            if self.heatmap_config['show_boxes']:
                processed_frame = self._draw_detections(heatmap_frame, detections)
            else:
                processed_frame = heatmap_frame  # Only show heatmap, no bounding boxes
            
            # Add information overlay
            if Config.DISPLAY_FPS:
                fps = self.camera_manager.get_camera_fps(camera_id)
                info_text = f"Camera {camera_id} | FPS: {fps:.1f} | Custom Heatmap"
                cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                time_text = f"Detection: {detection_time*1000:.1f}ms | Objects: {len(detections)}"
                cv2.putText(processed_frame, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                colormap_name = self._get_colormap_name(self.heatmap_config['colormap'])
                cv2.putText(processed_frame, f"Colormap: {colormap_name}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show bounding box status
                box_status = "ON" if self.heatmap_config['show_boxes'] else "OFF"
                cv2.putText(processed_frame, f"Boxes: {box_status}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['detection_time'] += detection_time
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return frame
    
    def _get_colormap_name(self, colormap: int) -> str:
        """Get colormap name."""
        colormap_names = {
            cv2.COLORMAP_JET: "JET", cv2.COLORMAP_HOT: "HOT", cv2.COLORMAP_VIRIDIS: "VIRIDIS",
            cv2.COLORMAP_PLASMA: "PLASMA", cv2.COLORMAP_INFERNO: "INFERNO", cv2.COLORMAP_MAGMA: "MAGMA",
            cv2.COLORMAP_TURBO: "TURBO", cv2.COLORMAP_RAINBOW: "RAINBOW", cv2.COLORMAP_OCEAN: "OCEAN",
            cv2.COLORMAP_COOL: "COOL"
        }
        return colormap_names.get(colormap, "UNKNOWN")
    
    def change_colormap(self, new_colormap: int) -> None:
        """Change colormap."""
        self.heatmap_config['colormap'] = new_colormap
        logger.info(f"Colormap changed to {self._get_colormap_name(new_colormap)}")
    
    def clear_heatmaps(self) -> None:
        """Clear all accumulated heatmaps."""
        for camera_id in self.heatmaps:
            self.heatmaps[camera_id].fill(0)
        logger.info("All heatmaps cleared")
    
    def toggle_bounding_boxes(self) -> None:
        """Toggle the display of bounding boxes on/off."""
        self.heatmap_config['show_boxes'] = not self.heatmap_config['show_boxes']
        status = "ON" if self.heatmap_config['show_boxes'] else "OFF"
        logger.info(f"Bounding boxes display: {status}")
        print(f"Bounding boxes: {status}")
    
    def initialize(self) -> bool:
        """Initialize the system."""
        try:
            # Load model
            if not self._load_model():
                return False
            
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
            
            logger.info(f"Custom heatmap system initialized with {len(active_cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def run(self) -> None:
        """Run the main detection loop."""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting custom heatmap detection system...")
        
        # Create windows
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Custom Heatmap"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
            x, y = self.camera_manager.get_window_position(camera_id)
            cv2.moveWindow(window_name, x, y)
        
        # Available colormaps
        colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA,
                    cv2.COLORMAP_INFERNO, cv2.COLORMAP_MAGMA, cv2.COLORMAP_TURBO, cv2.COLORMAP_RAINBOW]
        current_colormap_index = 0
        
        try:
            while self.running:
                frames = self.camera_manager.get_all_frames()
                if not frames:
                    time.sleep(0.01)
                    continue
                
                for camera_id, frame in frames.items():
                    start_time = time.time()
                    processed_frame = self.process_frame(frame, camera_id)
                    window_name = f"Camera {camera_id} - Custom Heatmap"
                    cv2.imshow(window_name, processed_frame)
                    self.stats['display_time'] += time.time() - start_time
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c'):  # Change colormap
                    current_colormap_index = (current_colormap_index + 1) % len(colormaps)
                    self.change_colormap(colormaps[current_colormap_index])
                elif key == ord('x'):  # Clear heatmaps
                    self.clear_heatmaps()
                elif key == ord('b'):  # Toggle bounding boxes
                    self.toggle_bounding_boxes()
                elif key == ord('s'):  # Save screenshot
                    self._save_screenshots(frames)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def _save_screenshots(self, frames: Dict[int, np.ndarray]) -> None:
        """Save screenshots."""
        timestamp = int(time.time())
        for camera_id, frame in frames.items():
            processed_frame = self.process_frame(frame, camera_id)
            filename = f"custom_heatmap_camera_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Screenshot saved: {filename}")
    
    def stop(self) -> None:
        """Stop the system."""
        self.running = False
        self.camera_manager.stop_all()
        cv2.destroyAllWindows()
        logger.info("Custom heatmap system stopped")

def main():
    """Main function."""
    print("🔥 Custom Heatmap Detection System")
    print("==================================")
    print("Controls:")
    print("- q/ESC: Quit")
    print("- c: Change colormap") 
    print("- x: Clear heatmaps")
    print("- b: Toggle bounding boxes ON/OFF")
    print("- s: Save screenshot")
    print()
    print("Bounding Box Toggle:")
    print("- Press 'b' to show only heatmap (no detection boxes)")
    print("- Press 'b' again to show both heatmap and detection boxes")
    print()
    
    system = HeatmapSystem()
    system.run()

if __name__ == "__main__":
    main()
