"""
Camera stream handler for RTSP and local cameras.
"""
import queue
import threading
import time
from typing import Optional, Union, Tuple
import cv2
import numpy as np
from loguru import logger

from config import Config
from utils import get_youtube_stream_url

class DemoStream:
    """Demo camera stream using sample images."""
    
    def __init__(self, camera_id: int):
        """Initialize demo stream."""
        self.camera_id = camera_id
        self.running = False
        self.frame_queue = queue.Queue(maxsize=Config.FRAME_BUFFER_SIZE)
        self.thread = None
        self.fps = 30.0  # Simulated FPS
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def start(self) -> bool:
        """Start the demo stream."""
        try:
            logger.info(f"Starting demo camera {self.camera_id}")
            self.running = True
            self.thread = threading.Thread(target=self._generate_frames, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            logger.error(f"Error starting demo camera {self.camera_id}: {e}")
            return False
    
    def _generate_frames(self) -> None:
        """Generate demo frames."""
        frame_interval = 1.0 / 30.0  # 30 FPS
        
        while self.running:
            try:
                # Create a demo frame with current timestamp
                frame = self._create_demo_frame()
                
                # Add frame to queue
                try:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame, block=False)
                    else:
                        # Remove old frame and add new one
                        try:
                            self.frame_queue.get(block=False)
                        except queue.Empty:
                            pass
                        self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
                
                time.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Error generating demo frame for camera {self.camera_id}: {e}")
                time.sleep(0.1)
    
    def _create_demo_frame(self) -> np.ndarray:
        """Create a demo frame with timestamp and moving objects."""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background color
        frame[:] = (50, 100, 50)  # Dark green background
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Demo Camera {self.camera_id}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, timestamp, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add a moving rectangle to simulate an object
        t = time.time()
        x = int((np.sin(t * 2) + 1) * (width - 100) / 2)
        y = int((np.cos(t * 1.5) + 1) * (height - 100) / 2)
        cv2.rectangle(frame, (x, y), (x + 80, y + 80), (0, 255, 255), -1)
        cv2.putText(frame, "Moving Object", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def stop(self) -> None:
        """Stop the demo stream."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except queue.Empty:
                break
        
        logger.info(f"Demo camera {self.camera_id} stopped")
    
    def is_running(self) -> bool:
        """Check if the demo stream is running."""
        return self.running

class CameraStream:
    """Thread-safe camera stream handler."""
    
    def __init__(self, stream_source: Union[int, str], camera_id: int):
        """
        Initialize camera stream.
        
        Args:
            stream_source: Camera index (int) or RTSP URL (str)
            camera_id: Unique identifier for the camera
        """
        self.stream_source = stream_source
        self.camera_id = camera_id
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=Config.FRAME_BUFFER_SIZE)
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def start(self) -> bool:
        """Start the camera stream."""
        try:
            stream_url = self.stream_source
            if isinstance(self.stream_source, str) and ('youtube.com' in self.stream_source or 'youtu.be' in self.stream_source):
                logger.info(f"Detected YouTube URL: {self.stream_source}")
                stream_url = get_youtube_stream_url(self.stream_source)
                if not stream_url:
                    logger.error(f"Could not get stream URL for YouTube video: {self.stream_source}")
                    return False
                logger.info(f"Using YouTube stream URL: {stream_url}")

            # Initialize video capture with RTSP optimizations
            if isinstance(stream_url, str) and stream_url.startswith('rtsp://'):
                logger.info(f"Starting RTSP camera {self.camera_id} with URL: {stream_url}")
                # For RTSP streams, use FFmpeg backend with optimized settings
                self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                
                # Set FFmpeg-specific options to prevent timeout issues
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_RECONNECT_TIMEOUT * 1000)
                
                # Additional FFmpeg options for better RTSP handling
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
                
                # Set video codec preference (remove problematic FOURCC setting for now)
                
            else:
                # For local cameras
                logger.info(f"Starting local camera {self.camera_id} with index/URL: {stream_url}")
                self.cap = cv2.VideoCapture(stream_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera stream: {self.stream_source}")
                return False
            
            # Configure capture properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.FRAME_BUFFER_SIZE)
            
            # Get camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera {self.camera_id} initialized: {width}x{height} @ {original_fps}fps")
              # Start capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            
            return True            
        except Exception as e:
            logger.error(f"Error starting camera {self.camera_id}: {e}")
            return False
    
    def _capture_frames(self) -> None:
        """Capture frames in a separate thread."""
        consecutive_failures = 0
        max_consecutive_failures = 5  # Reduced from 10 for faster recovery
        frame_timeout_count = 0
        max_frame_timeouts = 3  # Allow 3 frame timeouts before reconnecting
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                start_time = time.time()
                
                # Use grab() and retrieve() for better timeout control
                grabbed = self.cap.grab()
                
                if grabbed:
                    ret, frame = self.cap.retrieve()
                    
                    if ret and frame is not None:
                        # Reset failure counters on success
                        consecutive_failures = 0
                        frame_timeout_count = 0
                        
                        # Update FPS calculation
                        self.frame_count += 1
                        current_time = time.time()
                        if current_time - self.last_fps_time >= 1.0:
                            self.fps = self.frame_count / (current_time - self.last_fps_time)
                            self.frame_count = 0
                            self.last_fps_time = current_time
                        
                        # Add frame to queue (non-blocking)
                        try:
                            if not self.frame_queue.full():
                                self.frame_queue.put(frame, block=False)
                            else:
                                # Remove old frame and add new one
                                try:
                                    self.frame_queue.get(block=False)
                                except queue.Empty:
                                    pass
                                self.frame_queue.put(frame, block=False)
                        except queue.Full:
                            pass  # Skip frame if queue is full
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Failed to retrieve frame from camera {self.camera_id} (source: {self.stream_source}). "
                                     f"Consecutive failures: {consecutive_failures}")
                else:
                    consecutive_failures += 1
                    # Check if this might be a timeout issue
                    elapsed = time.time() - start_time
                    if elapsed > 25:  # If grab() took more than 25 seconds, likely a timeout
                        frame_timeout_count += 1
                        logger.warning(f"Frame grab timeout for camera {self.camera_id} (timeout {frame_timeout_count}/{max_frame_timeouts})")
                        
                        if frame_timeout_count >= max_frame_timeouts:
                            logger.error(f"Too many frame timeouts for camera {self.camera_id}. Attempting to reconnect...")
                            self._reconnect()
                            consecutive_failures = 0
                            frame_timeout_count = 0
                            continue
                    
                    logger.warning(f"Failed to grab frame from camera {self.camera_id} (source: {self.stream_source}). "
                                 f"Consecutive failures: {consecutive_failures}")
                
                # If too many consecutive failures, try to reconnect
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures for camera {self.camera_id}. Attempting to reconnect...")
                    if self._reconnect():
                        consecutive_failures = 0
                        frame_timeout_count = 0
                    else:
                        # If reconnection fails, wait longer before trying again
                        time.sleep(5.0)
                        consecutive_failures = 0
                
                # Small delay to prevent excessive CPU usage, but only if no timeout occurred
                elapsed = time.time() - start_time
                if elapsed < 1.0:  # Only sleep if the operation was fast
                    time.sleep(0.033)  # ~30fps max
                    
            except Exception as e:
                logger.error(f"Error capturing frame from camera {self.camera_id}: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the camera."""
        try:
            logger.info(f"Attempting to reconnect camera {self.camera_id} (source: {self.stream_source})")
            
            # Release current capture
            if self.cap:
                self.cap.release()
                self.cap = None
              # Wait a moment before reconnecting
            time.sleep(2.0)
            
            stream_url = self.stream_source
            if isinstance(self.stream_source, str) and ('youtube.com' in self.stream_source or 'youtu.be' in self.stream_source):
                logger.info(f"Re-getting stream for YouTube URL: {self.stream_source}")
                stream_url = get_youtube_stream_url(self.stream_source)
                if not stream_url:
                    logger.error(f"Could not get stream URL for YouTube video on reconnect: {self.stream_source}")
                    return False
                logger.info(f"Using new YouTube stream URL for reconnect: {stream_url}")

            # Re-initialize video capture with RTSP optimizations
            if isinstance(stream_url, str) and stream_url.startswith('rtsp://'):
                # For RTSP streams, use FFmpeg backend with optimized settings
                self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                
                # Set FFmpeg-specific options to prevent timeout issues
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_RECONNECT_TIMEOUT * 1000)
                
                # Additional FFmpeg options for better RTSP handling
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce latency
                
            else:
                # For local cameras
                self.cap = cv2.VideoCapture(stream_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to reconnect camera {self.camera_id}")
                return False
            
            # Configure capture properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.FRAME_BUFFER_SIZE)
            
            logger.info(f"Successfully reconnected camera {self.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error reconnecting camera {self.camera_id}: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the camera."""
        try:
            return self.frame_queue.get(block=False)
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Get current FPS of the camera stream."""
        return self.fps
    
    def stop(self) -> None:
        """Stop the camera stream."""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
          # Clear the frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(block=False)
            except queue.Empty:
                break
        
        logger.info(f"Camera {self.camera_id} stopped")
    
    def is_running(self) -> bool:
        """Check if the camera stream is running."""
        return self.running and self.cap is not None and self.cap.isOpened()

class MultiCameraManager:
    """Manager for multiple camera streams."""
    
    def __init__(self):
        """Initialize the multi-camera manager."""
        self.cameras = {}
        self.window_positions = {}
    
    def add_camera(self, stream_source: Union[int, str], camera_id: int) -> bool:
        """
        Add a camera stream.
        
        Args:
            stream_source: Camera index, RTSP URL, or 'demo' for demo mode
            camera_id: Unique identifier for the camera
            
        Returns:
            True if camera was added successfully
        """
        if camera_id in self.cameras:
            logger.warning(f"Camera {camera_id} already exists")
            return False
        
        # Check if demo mode is requested
        if stream_source == "demo":
            camera = DemoStream(camera_id)
        else:
            camera = CameraStream(stream_source, camera_id)
            
        if camera.start():
            self.cameras[camera_id] = camera
            self._calculate_window_position(camera_id)
            logger.info(f"Added camera {camera_id}: {stream_source}")
            return True
        else:
            logger.error(f"Failed to start camera {camera_id}: {stream_source}")
            return False
    
    def _calculate_window_position(self, camera_id: int) -> None:
        """Calculate window position for camera display."""
        # Arrange windows in a grid
        cols = 2  # Number of columns
        col = camera_id % cols
        row = camera_id // cols
        
        x = col * (Config.WINDOW_WIDTH + 50)
        y = row * (Config.WINDOW_HEIGHT + 100)
        
        self.window_positions[camera_id] = (x, y)
    
    def get_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get frame from specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_frame()
        return None
    
    def get_all_frames(self) -> dict:
        """Get frames from all cameras."""
        frames = {}
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is not None:
                frames[camera_id] = frame
        return frames
    
    def get_camera_fps(self, camera_id: int) -> float:
        """Get FPS for specific camera."""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get_fps()
        return 0.0
    
    def get_window_position(self, camera_id: int) -> Tuple[int, int]:
        """Get window position for camera display."""
        return self.window_positions.get(camera_id, (0, 0))
    
    def stop_camera(self, camera_id: int) -> None:
        """Stop specific camera."""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            if camera_id in self.window_positions:
                del self.window_positions[camera_id]
    
    def stop_all(self) -> None:
        """Stop all cameras."""
        for camera in self.cameras.values():
            camera.stop()
        self.cameras.clear()
        self.window_positions.clear()
    
    def get_active_cameras(self) -> list:
        """Get list of active camera IDs."""
        return [cid for cid, camera in self.cameras.items() if camera.is_running()]
