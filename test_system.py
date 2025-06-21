#!/usr/bin/env python3
"""
Test script for the real-time object detection system.
This script performs basic functionality tests without requiring cameras.
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from config import Config
        from detector import ObjectDetector
        from camera_stream import CameraStream, MultiCameraManager
        from utils import check_system_requirements, create_test_image
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_system_requirements():
    """Test system requirements."""
    print("\nTesting system requirements...")
    try:
        from utils import check_system_requirements, print_system_info
        print_system_info()
        return True
    except Exception as e:
        print(f"❌ System requirements check failed: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading."""
    print("\nTesting YOLO model...")
    try:
        from detector import ObjectDetector
        detector = ObjectDetector()
        print(f"✅ YOLO model loaded: {len(detector.class_names)} classes")
        return True
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False

def test_detection_on_test_image():
    """Test object detection on a synthetic test image."""
    print("\nTesting object detection...")
    try:
        from detector import ObjectDetector
        from utils import create_test_image
        
        # Create detector and test image
        detector = ObjectDetector()
        test_img = create_test_image()
        
        # Run detection
        detections = detector.detect(test_img)
        
        # Draw results
        result_img = detector.draw_detections(test_img, detections, camera_id=0)
        
        print(f"✅ Detection successful: {len(detections)} objects detected")
        
        # Save test result
        cv2.imwrite("test_detection_result.jpg", result_img)
        print("✅ Test result saved as 'test_detection_result.jpg'")
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_camera_stream():
    """Test camera stream (without actual camera)."""
    print("\nTesting camera stream (mock)...")
    try:
        from camera_stream import MultiCameraManager
        manager = MultiCameraManager()
        print("✅ Camera manager created successfully")
        return True
    except Exception as e:
        print(f"❌ Camera stream test failed: {e}")
        return False

def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        from config import Config
        print(f"✅ Model path: {Config.MODEL_NAME}")
        print(f"✅ Confidence threshold: {Config.MODEL_CONFIDENCE}")
        print(f"✅ Device: {Config.DEVICE}")
        print(f"✅ GPU available: {Config.validate_gpu_availability()}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Real-time Object Detection System - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_system_requirements,
        test_config,
        test_yolo_model,
        test_detection_on_test_image,
        test_camera_stream,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo run the application:")
        print("  • Execute: run.bat")
        print("  • Or run: uv run python main.py")
        print("  • Or use PowerShell: .\\run.ps1")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
