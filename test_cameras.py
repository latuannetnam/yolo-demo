#!/usr/bin/env python3
"""
Test script to detect and test available cameras.
"""
import cv2
import os
from loguru import logger

def create_tcp_video_capture(rtsp_url: str) -> cv2.VideoCapture:
    """Create a VideoCapture instance configured for TCP transport."""
    try:
        # Import config to get transport settings
        from config import Config
        
        # Store original FFmpeg options
        original_env = os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', '')
        
        # Configure FFmpeg options for TCP transport
        transport = Config.RTSP_TRANSPORT.lower()
        stimeout = Config.RTSP_STIMEOUT
        
        # Build FFmpeg options string
        ffmpeg_options = f'rtsp_transport;{transport}|stimeout;{stimeout}|timeout;30000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;10'
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = ffmpeg_options
        
        logger.debug(f"Setting FFmpeg options: {ffmpeg_options}")
        
        # Create VideoCapture with FFmpeg backend
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Set OpenCV-specific properties
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, Config.RTSP_RECONNECT_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.FRAME_BUFFER_SIZE)
        
        # Restore original environment
        if original_env:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = original_env
        else:
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
        
        logger.info(f"Created VideoCapture with {transport.upper()} transport for: {rtsp_url}")
        return cap
        
    except Exception as e:
        logger.error(f"Error creating TCP VideoCapture for {rtsp_url}: {e}")
        # Fallback to standard VideoCapture
        return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

def test_camera_index(index: int) -> bool:
    """Test if a camera index is available."""
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret and frame is not None
        return False
    except Exception as e:
        logger.error(f"Error testing camera {index}: {e}")
        return False

def find_available_cameras(max_cameras: int = 1) -> list:
    """Find all available camera indices."""
    available_cameras = []
    
    for i in range(max_cameras):
        if test_camera_index(i):
            available_cameras.append(i)
            logger.info(f"Camera {i} is available")
        else:
            logger.debug(f"Camera {i} is not available")
    
    return available_cameras

def test_rtsp_stream(rtsp_url: str) -> bool:
    """Test if an RTSP stream is accessible using configured transport protocol."""
    try:
        logger.info(f"Testing RTSP stream: {rtsp_url}")
        
        # Use the TCP-configured VideoCapture
        cap = create_tcp_video_capture(rtsp_url)
        
        if cap.isOpened():
            logger.info(f"RTSP stream opened successfully: {rtsp_url}")
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                logger.info(f"Successfully read frame from RTSP stream: {rtsp_url}")
                return True
            else:
                logger.warning(f"Failed to read frame from RTSP stream: {rtsp_url}")
                return False
        else:
            logger.error(f"Failed to open RTSP stream: {rtsp_url}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing RTSP stream {rtsp_url}: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing camera availability...")
    
    # Show transport configuration
    from config import Config
    logger.info(f"RTSP Transport configured: {Config.RTSP_TRANSPORT}")
    logger.info(f"RTSP Timeout: {Config.RTSP_TIMEOUT}s")
    logger.info(f"RTSP Stream Timeout: {Config.RTSP_STIMEOUT}μs")
    
    # Test local cameras
    available_cameras = find_available_cameras()
    logger.info(f"Available local cameras: {available_cameras}")
    
    # Test RTSP stream from config
    streams = Config.get_stream_config()
    
    for stream in streams:
        if stream.startswith("rtsp://"):
            logger.info(f"Testing RTSP stream with {Config.RTSP_TRANSPORT.upper()} transport: {stream}")
            if test_rtsp_stream(stream):
                logger.info(f"RTSP stream {stream} is accessible")
            else:
                logger.warning(f"RTSP stream {stream} is not accessible")
        else:
            try:
                camera_index = int(stream)
                if camera_index in available_cameras:
                    logger.info(f"Camera {camera_index} is available")
                else:
                    logger.warning(f"Camera {camera_index} is not available")
            except ValueError:
                logger.error(f"Invalid camera stream format: {stream}")
