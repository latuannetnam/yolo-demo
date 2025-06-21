"""
Real-time instance segmentation system with multi-camera support using Ultralytics Solutions.
"""
import cv2
import time
import signal
import sys
from typing import Dict, Any
import numpy as np
from loguru import logger
from ultralytics import solutions

from config import Config
from camera_stream import MultiCameraManager

# Setup logging
# logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="INFO")


class SolutionSegmentationSystem:
    """Main instance segmentation system using Ultralytics Solutions."""
    
    def __init__(self):
        """Initialize the segmentation system."""
        self.isegment = None
        self.camera_manager = MultiCameraManager()
        self.running = False
        self.stats = {
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
        """Initialize the segmentation system components."""
        try:
            # Initialize instance segmentation using Ultralytics Solutions
            logger.info("Initializing instance segmentation with Ultralytics Solutions...")
            model_path = Config.get_full_segmentation_model_name()
            
            self.isegment = solutions.InstanceSegmentation(
                show=False,  # We'll handle display ourselves
                model=model_path,
                # classes=[0, 2],  # Optional: segment specific classes (person, car)
                conf=Config.MODEL_CONFIDENCE,  # confidence threshold for model
                iou=Config.MODEL_IOU_THRESHOLD,  # IOU threshold for model
                verbose=False,
            )
            
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
        Process a single frame with instance segmentation.
        
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
            
            # Check if segmentation is initialized
            if self.isegment is None:
                logger.error("Instance segmentation not initialized")
                return frame
            
            # Perform instance segmentation
            start_time = time.time()
            results = self.isegment(frame)
            segmentation_time = time.time() - start_time
            
            # Get the processed frame
            if results is not None and hasattr(results, 'plot_im') and results.plot_im is not None:
                processed_frame = results.plot_im
            else:
                # Fallback to original frame if results don't have plot_im
                processed_frame = frame
            
            # Add FPS and camera info
            if Config.DISPLAY_FPS:
                fps = self.camera_manager.get_camera_fps(camera_id)
                info_text = f"Camera {camera_id} | FPS: {fps:.1f}"
                cv2.putText(
                    processed_frame,
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
                    processed_frame,
                    time_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['segmentation_time'] += segmentation_time
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return frame
    
    def run(self) -> None:
        """Run the main segmentation loop."""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting instance segmentation system...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Instance Segmentation"
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
                    window_name = f"Camera {camera_id} - Instance Segmentation"
                    cv2.imshow(window_name, processed_frame)
                    
                    display_time = time.time() - start_time
                    self.stats['display_time'] += display_time
                
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
            filename = f"solution_segmentation_screenshot_camera_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Solution segmentation screenshot saved: {filename}")
    
    def _reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_frames': 0,
            'segmentation_time': 0.0,
            'display_time': 0.0
        }
        logger.info("Statistics reset")
    
    def stop(self) -> None:
        """Stop the segmentation system."""
        self.running = False
        
        # Stop all cameras
        self.camera_manager.stop_all()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.stats['total_frames'] > 0:
            avg_segmentation_time = self.stats['segmentation_time'] / self.stats['total_frames']
            avg_display_time = self.stats['display_time'] / self.stats['total_frames']
            
            logger.info("=== Final Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average segmentation time: {avg_segmentation_time*1000:.2f}ms")
            logger.info(f"Average display time: {avg_display_time*1000:.2f}ms")
        
        logger.info("Solution instance segmentation system stopped")

def main():
    """Main function."""
    print("🎯 Real-time Instance Segmentation System (Ultralytics Solutions) v1.0")
    print("======================================================================")
    print("Features:")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time YOLOv11 instance segmentation using Ultralytics Solutions")
    print("- GPU acceleration (if available)")
    print("- Segmentation masks with advanced visualization")
    print("- Separate display windows for each camera")
    print()
    print("Controls:")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save screenshots")
    print("- Press 'r' to reset statistics")
    print()
    
    # Create and run the segmentation system
    system = SolutionSegmentationSystem()
    
    try:
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()