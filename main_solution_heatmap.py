import cv2
import time
import signal
import sys
from typing import Optional
import numpy as np
from loguru import logger

from ultralytics import solutions
from config import Config
from camera_stream import MultiCameraManager

# Setup logging
logger.add("logs/heatmap_app.log", rotation="10 MB", retention="7 days", level="INFO")

class HeatmapSystem:
    """Heatmap detection system with camera stream support."""
    
    def __init__(self):
        """Initialize the heatmap system."""
        self.camera_manager = MultiCameraManager()
        self.heatmap = None
        self.running = False
        self.video_writer = None
        self.stats = {
            'total_frames': 0,
            'processing_time': 0.0,
            'display_time': 0.0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping heatmap system...")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize the heatmap system components."""
        try:
            # Get model path from config
            model_path = Config.get_full_model_name()
            logger.info(f"Using model: {model_path}")
            
            # Initialize heatmap object
            self.heatmap = solutions.Heatmap(
                show=False,  # display the output
                model=model_path,  # path to the YOLO model file from config
                colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap,
                show_in=False,  # Flag to control whether to display the in counts on the video stream
                show_out=False,  # Flag to control whether to display the out counts on the video stream
                # region=region_points,  # object counting with heatmaps, you can pass region_points
                # classes=[2],  # generate heatmap for specific classes i.e person and car.
                conf=Config.MODEL_CONFIDENCE,  # confidence threshold for model
                iou=Config.MODEL_IOU_THRESHOLD,  # IOU threshold for model
                verbose=False,  # suppress verbose output
            )
            
            # Setup cameras
            logger.info("Setting up cameras...")
            streams = Config.get_validated_streams()
            
            for i, stream in enumerate(streams):
                if self.camera_manager.add_camera(stream, i):
                    logger.info(f"Camera {i} added successfully")
                else:
                    logger.error(f"Failed to add camera {i}: {stream}")
            
            active_cameras = self.camera_manager.get_active_cameras()
            if not active_cameras:
                logger.error("No cameras available. Exiting.")
                return False
            
            logger.info(f"Heatmap system initialized with {len(active_cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize heatmap system: {e}")
            return False

    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> Optional[np.ndarray]:
        """
        Process a single frame with heatmap detection.
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            Processed frame with heatmap or None if processing fails
        """
        try:
            if self.heatmap is None:
                return frame
            
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            # Perform heatmap processing
            start_time = time.time()
            results = self.heatmap(frame)
            processing_time = time.time() - start_time
            
            # Get the processed frame
            if results is not None and hasattr(results, 'plot_im'):
                processed_frame = results.plot_im
            else:
                processed_frame = frame
            
            # Add information overlay
            if Config.DISPLAY_FPS:
                # Get camera FPS
                fps = self.camera_manager.get_camera_fps(camera_id)
                
                # Get model information
                model_name = Config.MODEL_NAME
                model_path = Config.get_full_model_name()
                model_type = "TensorRT" if model_path.endswith('.engine') else "YOLO"
                
                # Add camera and FPS info
                info_text = f"Camera {camera_id} | FPS: {fps:.1f} | Heatmap Detection"
                cv2.putText(
                    processed_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Add processing time info
                time_text = f"Processing: {processing_time*1000:.1f}ms"
                cv2.putText(
                    processed_frame,
                    time_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                
                # Add model information
                model_text = f"Model: {model_name} ({model_type})"
                cv2.putText(
                    processed_frame,
                    model_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )
                
                # Add confidence threshold info
                conf_text = f"Confidence: {Config.MODEL_CONFIDENCE}"
                cv2.putText(
                    processed_frame,
                    conf_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1
                )
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['processing_time'] += processing_time
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {e}")
            return frame
    
    def run(self) -> None:
        """Main processing loop."""
        if not self.initialize():
            logger.error("Failed to initialize heatmap system")
            return
        
        self.running = True
        logger.info("Starting heatmap processing...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Heatmap Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
            
            # Position windows
            x, y = self.camera_manager.get_window_position(camera_id)
            cv2.moveWindow(window_name, x, y)
        
        try:
            while self.running:
                active_cameras = self.camera_manager.get_active_cameras()
                
                if not active_cameras:
                    logger.warning("No active cameras available")
                    time.sleep(1)
                    continue
                
                for camera_id in active_cameras:
                    frame = self.camera_manager.get_frame(camera_id)
                    if frame is not None:
                        start_time = time.time()
                        
                        # Process frame with heatmap
                        processed_frame = self.process_frame(frame, camera_id)
                        
                        if processed_frame is not None:
                            # Display the frame
                            window_name = f"Camera {camera_id} - Heatmap Detection"
                            cv2.imshow(window_name, processed_frame)
                            
                            # Optional: Save processed frame to video file
                            if self.video_writer is not None:
                                self.video_writer.write(processed_frame)
                        
                        display_time = time.time() - start_time
                        self.stats['display_time'] += display_time
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    logger.info("Quit key pressed")
                    break
                elif key == ord('r'):  # 'r' key to reset stats
                    self._reset_stats()
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the heatmap system."""
        logger.info("Stopping heatmap system...")
        self.running = False
        
        # Stop camera manager
        if self.camera_manager:
            self.camera_manager.stop_all()
        
        # Release video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Close all windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.stats['total_frames'] > 0:
            avg_processing_time = self.stats['processing_time'] / self.stats['total_frames']
            avg_display_time = self.stats['display_time'] / self.stats['total_frames']
            
            logger.info("=== Final Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average processing time: {avg_processing_time*1000:.2f}ms")
            logger.info(f"Average display time: {avg_display_time*1000:.2f}ms")
        
        logger.info("Heatmap system stopped")    
    

def main():
    """Main function to run the heatmap system."""
    logger.info("Starting Heatmap Detection System")
    
    try:
        # Create and run the heatmap system
        heatmap_system = HeatmapSystem()
        heatmap_system.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    
    logger.info("Heatmap Detection System shutdown complete")

if __name__ == "__main__":
    main()