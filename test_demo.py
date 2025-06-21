#!/usr/bin/env python3
"""
Test demo camera stream.
"""
import cv2
import time
from loguru import logger
from camera_stream import DemoStream

def test_demo_stream():
    """Test the demo camera stream."""
    logger.info("Testing demo camera stream...")
    
    # Create demo stream
    demo = DemoStream(camera_id=0)
    
    if not demo.start():
        logger.error("Failed to start demo stream")
        return
    
    logger.info("Demo stream started successfully")
    
    # Test for a few seconds
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 5.0:  # Test for 5 seconds
        frame = demo.get_frame()
        if frame is not None:
            frame_count += 1
            logger.info(f"Received frame {frame_count}, shape: {frame.shape}")
        time.sleep(0.1)
    
    demo.stop()
    logger.info(f"Demo test completed. Received {frame_count} frames in 5 seconds")

if __name__ == "__main__":
    test_demo_stream()
