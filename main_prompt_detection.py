"""
Real-time prompt-based object detection system with multi-camera support.
"""
import cv2
import time
import signal
import sys
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger

from config import Config
from detector import PromptDetector  # Use PromptDetector
from camera_stream import MultiCameraManager

# Setup logging
# logger.add("logs/prompt_app.log", rotation="10 MB", retention="7 days", level="INFO")

class PromptDetectionSystem:
    """Main prompt-based object detection system."""
    
    def __init__(self, initial_prompts: Optional[List[str]] = None):
        """Initialize the detection system."""
        # self.detector = None
        self.camera_manager = MultiCameraManager()
        self.running = False
        self.stats = {
            'total_frames': 0,
            'detection_time': 0,
            'display_time': 0
        }
        self.initial_prompts = initial_prompts if initial_prompts else ["car"] # Default prompts
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize the detection system components."""
        try:
            # Initialize prompt-based object detector
            logger.info("Initializing prompt-based object detector...")
            self.detector = PromptDetector()
            
            # Set initial prompts
            if self.initial_prompts:
                logger.info(f"Setting initial prompts: {self.initial_prompts}")
                if not self.detector.set_prompts(self.initial_prompts):
                    logger.error("Failed to set initial prompts. Please check model and prompts.")
                    # return False # Allow to continue without prompts, user can set them later
            
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
            
            logger.info(f"System initialized with {len(active_cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> np.ndarray:
        """
        Process a single frame with prompt-based object detection.
        
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
            
            # Perform object detection (and segmentation if model supports it)
            start_time = time.time()
            # Using detect_and_segment for YOLOE models as they usually support it
            # If your YOLOE model is detection-only, you might want to use self.detector.detect(frame)
            detections = self.detector.detect_and_segment(frame)
            detection_time = time.time() - start_time
            
            # Draw detections (and segmentations if available)
            # Using draw_segmentations for YOLOE models
            frame_with_detections = self.detector.draw_segmentations(frame, detections, camera_id)
            
            # Add FPS and camera info
            if Config.DISPLAY_FPS:
                fps = self.camera_manager.get_camera_fps(camera_id)
                current_prompts_str = ", ".join(self.detector.get_current_prompts())
                info_text = f"Cam {camera_id} | FPS: {fps:.1f} | Objects: {len(detections)}"
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
                    (10, 60), # Position below FPS info
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0), # Cyan color for prompt text
                    1
                )
                
                # Add detection time info
                time_text = f"Detection: {detection_time*1000:.1f}ms"
                cv2.putText(
                    frame_with_detections,
                    time_text,
                    (10, 90), # Position below prompt text
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['detection_time'] += detection_time
            
            return frame_with_detections
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return frame
    
    def run(self) -> None:
        """Run the main detection loop."""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting prompt-based object detection system...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Prompt Detection"
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
                    window_name = f"Camera {camera_id} - Prompt Detection"
                    cv2.imshow(window_name, processed_frame)
                    
                    display_time = time.time() - start_time
                    self.stats['display_time'] += display_time
                
                # Check for exit key or prompt change
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # 's' key to save screenshot
                    self._save_screenshots(frames)
                elif key == ord('r'):  # 'r' key to reset stats
                    self._reset_stats()
                elif key == ord('p'): # 'p' key to change prompts
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
            current_prompts_str = ", ".join(self.detector.get_current_prompts())
            new_prompts_str = input(f"Enter new prompts (comma-separated, current: [{current_prompts_str}]): ")
            if new_prompts_str:
                new_prompts = [p.strip() for p in new_prompts_str.split(',') if p.strip()]
                if new_prompts:
                    logger.info(f"Changing prompts to: {new_prompts}")
                    if self.detector.set_prompts(new_prompts):
                        logger.info("Prompts updated successfully.")
                    else:
                        logger.error("Failed to update prompts.")
                else:
                    logger.info("No valid prompts entered.")
            else:
                logger.info("Prompt change cancelled or no input.")
        except Exception as e:
            logger.error(f"Error changing prompts: {e}")

    def _save_screenshots(self, frames: Dict[int, np.ndarray]) -> None:
        """Save screenshots of current frames."""
        timestamp = int(time.time())
        for camera_id, frame in frames.items():
            # Re-process frame to include latest detections and info
            processed_frame = self.process_frame(frame, camera_id) 
            filename = f"prompt_screenshot_cam_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Screenshot saved: {filename}")
    
    def _reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_frames': 0,
            'detection_time': 0,
            'display_time': 0
        }
        logger.info("Statistics reset")
    
    def stop(self) -> None:
        """Stop the detection system."""
        self.running = False
        
        # Stop all cameras
        self.camera_manager.stop_all()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.stats['total_frames'] > 0:
            avg_detection_time = self.stats['detection_time'] / self.stats['total_frames']
            avg_display_time = self.stats['display_time'] / self.stats['total_frames']
            
            logger.info("=== Final Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average detection time: {avg_detection_time*1000:.2f}ms")
            logger.info(f"Average display time: {avg_display_time*1000:.2f}ms")
        
        logger.info("Prompt-based object detection system stopped")

def main():
    """Main function."""
    print("🚀 Real-time Prompt-Based Object Detection System (YOLOE) v1.0")
    print("=============================================================")
    print("Features:")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time YOLOE object detection with text prompts")
    print("- GPU acceleration (if available)")
    print("- Separate display windows for each camera")
    print("- Dynamic prompt changing")
    print("- Initial prompts configured via INITIAL_PROMPTS environment variable")
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

    # Create and run the detection system
    system = PromptDetectionSystem(initial_prompts=initial_prompts_list)
    
    try:
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
