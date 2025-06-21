"""
Real-time traffic light violation detection system.
"""
import cv2
import time
import signal
import sys
from typing import Dict, Any
import numpy as np
from loguru import logger

from config import Config
from detector import ObjectDetector
from camera_stream import MultiCameraManager
from traffic_light_detector import TrafficLightViolationDetector

class TrafficViolationSystem:
    """Main traffic light violation detection system."""
    
    def __init__(self):
        """Initialize the traffic violation system."""
        self.detector = None
        self.violation_detector = TrafficLightViolationDetector()
        self.camera_manager = MultiCameraManager()
        self.running = False
        self.stats = {
            'total_frames': 0,
            'detection_time': 0.0,
            'violations_detected': 0
        }
        
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
            # Initialize object detector
            logger.info("Initializing object detector...")
            self.detector = ObjectDetector()
            
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
        Process a single frame with object detection and violation monitoring.
        
        Args:
            frame: Input frame
            camera_id: Camera identifier
            
        Returns:
            Processed frame with detections and violation indicators
        """
        try:
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            # Setup violation zones if not already done
            if not self.violation_detector.violation_zones:
                self.violation_detector.setup_violation_zones_from_image(frame.shape[:2])
            
            # Perform object detection
            start_time = time.time()
            detections = []
            if self.detector is not None:
                detections = self.detector.detect(frame)
            detection_time = time.time() - start_time
            
            # Draw basic detections
            frame_with_detections = frame
            if self.detector is not None:
                frame_with_detections = self.detector.draw_detections(frame, detections, camera_id)
            
            # Draw violation zones
            frame_with_detections = self.violation_detector.draw_violation_zones(frame_with_detections)
            
            # Detect traffic light states and draw them
            frame_with_detections, traffic_light_states = self.violation_detector.draw_traffic_light_states(
                frame_with_detections, detections
            )
            
            # Update vehicle tracking
            self.violation_detector.update_vehicle_tracks(detections)
            
            # Check for violations
            violations = self.violation_detector.check_violations(traffic_light_states)
            
            # Draw violations
            if violations:
                frame_with_detections = self.violation_detector.draw_violations(frame_with_detections, violations)
                self.stats['violations_detected'] += len(violations)
                
                # Log violations
                for violation in violations:
                    logger.warning(f"Traffic violation: {violation['vehicle_class']} at {violation['position']} in {violation['zone_name']}")
            
            # Add system info
            if Config.DISPLAY_FPS:
                fps = self.camera_manager.get_camera_fps(camera_id)
                info_text = f"Camera {camera_id} | FPS: {fps:.1f} | Objects: {len(detections)}"
                cv2.putText(
                    frame_with_detections,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Add detection time info
                time_text = f"Detection: {detection_time*1000:.1f}ms"
                cv2.putText(
                    frame_with_detections,
                    time_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
                
                # Add violation count
                violation_text = f"Violations: {self.stats['violations_detected']}"
                cv2.putText(
                    frame_with_detections,
                    violation_text,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
                
                # Add traffic light status
                if traffic_light_states:
                    states_text = "Lights: " + ", ".join([state.value for state in traffic_light_states.values()])
                    cv2.putText(
                        frame_with_detections,
                        states_text,
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )
            
            # Update stats
            self.stats['total_frames'] += 1
            self.stats['detection_time'] = self.stats['detection_time'] + detection_time
            
            return frame_with_detections
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
            return frame
    
    def run(self) -> None:
        """Run the main detection loop."""
        if not self.initialize():
            return
        
        self.running = True
        logger.info("Starting traffic violation detection system...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Traffic Violation Detection"
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
                    window_name = f"Camera {camera_id} - Traffic Violation Detection"
                    cv2.imshow(window_name, processed_frame)
                    
                    display_time = time.time() - start_time
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # 's' key to save screenshot
                    self._save_screenshots(frames)
                elif key == ord('r'):  # 'r' key to reset stats
                    self._reset_stats()
                elif key == ord('z'):  # 'z' key to reset violation zones
                    self._reset_violation_zones()
        
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
            filename = f"violation_screenshot_camera_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Screenshot saved: {filename}")
    
    def _reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_frames': 0,
            'detection_time': 0.0,
            'violations_detected': 0
        }
        logger.info("Statistics reset")
    
    def _reset_violation_zones(self) -> None:
        """Reset violation zones and vehicle tracks."""
        self.violation_detector.violation_zones = []
        self.violation_detector.vehicle_tracks = {}
        self.violation_detector.violations = []
        logger.info("Violation zones and tracks reset")
    
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
            
            logger.info("=== Final Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average detection time: {avg_detection_time*1000:.2f}ms")
            logger.info(f"Total violations detected: {self.stats['violations_detected']}")
        
        logger.info("Traffic violation detection system stopped")

def main():
    """Main function."""
    print("🚦 Traffic Light Violation Detection System v1.0")
    print("===============================================")
    print("Features:")
    print("- Real-time traffic light state detection")
    print("- Vehicle tracking and violation monitoring")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time YOLOv11 object detection")
    print("- GPU acceleration (if available)")
    print()
    print("Detection Capabilities:")
    print("- Traffic light color detection (Red/Yellow/Green)")
    print("- Vehicle movement tracking")
    print("- Red light violation detection")
    print("- Violation zone monitoring")
    print()
    print("Controls:")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save screenshots")
    print("- Press 'r' to reset statistics")
    print("- Press 'z' to reset violation zones")
    print()
    
    # Create and run the detection system
    system = TrafficViolationSystem()
    
    try:
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
