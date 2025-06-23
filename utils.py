"""
Utility functions for the object detection system.
"""
import os
import sys
import time
import psutil
from typing import Dict, Any, Tuple
import cv2
import numpy as np
from loguru import logger
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

def get_youtube_stream_url(youtube_url: str) -> str | None:
        """
        Get the stream URL for a given YouTube URL.
        Tries to get the best 1080p mp4 stream, falls back to best available.
        """
        if not yt_dlp:
            logger.warning("yt-dlp not found, cannot process YouTube URLs. Please install with 'pip install yt-dlp'")
            return None
        logger.info(f"Getting stream URL for {youtube_url}")
        
        ydl_opts = {
            'quiet': True,
            'format': 'bestvideo[ext=mp4][height<=1080][vcodec=h264]+bestaudio[ext=m4a]/best[ext=mp4][vcodec=h264]/best',
            # 'format': 'bestvideo[ext=mp4][height<=1080][vcodec=h264]+bestaudio[ext=m4a]',
            # 'format': 'bestvideo[vcodec^=h264][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=h264][height<=1080]',
            'noplaylist': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)
                
                if not info_dict:
                    logger.error("yt-dlp failed to extract info.")
                    return None
                
                # if 'url' in info_dict:
                #     return info_dict['url']
                
                formats = info_dict.get('formats', [info_dict])
                for f in reversed(formats):
                    if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('url') and f.get('height') == 1080:
                        logger.info(f"Found mp4 stream: {f.get('format_note')}")
                        return f['url']
                
                logger.warning("No direct mp4 stream found. Falling back to the first available stream URL.")
                if formats and formats[0].get('url'):
                    return formats[0]['url']
                
                logger.error("Could not find any stream URL.")
                return None

        except Exception as e:
            logger.error(f"yt-dlp failed to get stream URL: {e}")
            return None

def check_system_requirements() -> Dict[str, Any]:
    """
    Check system requirements and capabilities.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'python_version': sys.version,
        'opencv_version': cv2.__version__,
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
        'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        'gpu_available': False,
        'gpu_name': None,
        'cuda_version': None
    }    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = "Available"  # Simplified to avoid version detection issues
    except ImportError:
        pass
    
    return info

def print_system_info():
    """Print system information and requirements check."""
    print("=== System Information ===")
    info = check_system_requirements()
    
    print(f"Python Version: {info['python_version']}")
    print(f"OpenCV Version: {info['opencv_version']}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"Total Memory: {info['memory_total']:.1f} GB")
    print(f"Available Memory: {info['memory_available']:.1f} GB")
    
    if info['gpu_available']:
        print(f"GPU: {info['gpu_name']}")
        print(f"CUDA Version: {info['cuda_version']}")
        print("✅ GPU acceleration available")
    else:
        print("⚠️  GPU not available, using CPU")
    
    print()

def download_yolo_model(model_name: str = "yolo11n.pt") -> bool:
    """
    Download YOLO model if it doesn't exist.
    
    Args:
        model_name: Name of the YOLO model to download
        
    Returns:
        True if model is available
    """
    if os.path.exists(model_name):
        logger.info(f"Model {model_name} already exists")
        return True
    
    try:
        from ultralytics import YOLO
        logger.info(f"Downloading YOLO model: {model_name}")
        
        # This will automatically download the model
        model = YOLO(model_name)
        logger.info(f"Model {model_name} downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        return False

def validate_rtsp_url(url: str) -> bool:
    """
    Validate RTSP URL format.
    
    Args:
        url: RTSP URL to validate
        
    Returns:
        True if URL format is valid
    """
    if not isinstance(url, str):
        return False
    
    return url.startswith('rtsp://') and len(url) > 9

def test_camera_connection(source, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test camera connection.
    
    Args:
        source: Camera source (index or RTSP URL)
        timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (success, message)
    """
    try:
        cap = cv2.VideoCapture(source)
        
        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        
        if not cap.isOpened():
            return False, f"Failed to open camera: {source}"
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, f"Failed to read frame from: {source}"
        
        h, w = frame.shape[:2]
        return True, f"Camera connected: {w}x{h}"
        
    except Exception as e:
        return False, f"Camera test error: {e}"

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a test image for testing detection without camera.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Test image with geometric shapes
    """
    # Create a test image with some shapes
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(height):
        intensity = int(50 + (y / height) * 50)
        img[y, :] = [intensity, intensity//2, intensity//3]
    
    # Draw some shapes that might be detected as objects
    # Rectangle (might be detected as a car/truck)
    cv2.rectangle(img, (100, 200), (250, 300), (0, 255, 0), -1)
    
    # Circle (might be detected as a ball/sports ball)
    cv2.circle(img, (400, 150), 50, (255, 255, 0), -1)
    
    # Another rectangle
    cv2.rectangle(img, (450, 250), (550, 350), (0, 0, 255), -1)
    
    # Add some text
    cv2.putText(img, "TEST IMAGE", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def log_performance_metrics(stats: Dict[str, Any]):
    """
    Log performance metrics to file.
    
    Args:
        stats: Performance statistics dictionary
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open("performance_log.txt", "a") as f:
            f.write(f"\n=== {timestamp} ===\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    except Exception as e:
        logger.error(f"Failed to log performance metrics: {e}")

def cleanup_old_screenshots(max_age_hours: int = 24):
    """
    Clean up old screenshot files.
    
    Args:
        max_age_hours: Maximum age of screenshots to keep
    """
    try:
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        for filename in os.listdir("."):
            if filename.startswith("screenshot_") and filename.endswith(".jpg"):
                file_path = os.path.join(".", filename)
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"Removed old screenshot: {filename}")
                    
    except Exception as e:
        logger.error(f"Failed to cleanup screenshots: {e}")

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string
    """
    value = float(bytes_value)  # Convert to float for division
    for unit in ['B', 'KB', 'MB', 'GB']:
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"

def get_gpu_memory_usage() -> Dict[str, Any]:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary with GPU memory info
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'allocated': allocated,
                'allocated_mb': allocated / 1024**2,
                'cached': cached,
                'cached_mb': cached / 1024**2,
                'total': total,
                'total_mb': total / 1024**2,
                'free_mb': (total - allocated) / 1024**2
            }
    except ImportError:
        pass
    
    return {}

def create_robust_rtsp_connection(rtsp_url: str, timeout_ms: int = 30000) -> cv2.VideoCapture:
    """
    Create a robust RTSP connection with optimized FFmpeg settings.
    
    Args:
        rtsp_url: RTSP stream URL
        timeout_ms: Timeout in milliseconds (default 30 seconds)
        
    Returns:
        cv2.VideoCapture object with optimized settings
    """
    try:
        # Create VideoCapture with FFmpeg backend
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Set comprehensive timeout and buffer settings
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms + 15000)  # Extra time for reading
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
        
        # Additional FFmpeg options for better RTSP handling
        # These settings help with network stability and stream compatibility
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
        
        logger.info(f"Created RTSP connection with {timeout_ms}ms timeout for: {rtsp_url}")
        return cap
        
    except Exception as e:
        logger.error(f"Failed to create RTSP connection for {rtsp_url}: {e}")
        # Fallback to standard connection
        return cv2.VideoCapture(rtsp_url)

def diagnose_rtsp_connection(rtsp_url: str) -> Dict[str, Any]:
    """
    Diagnose RTSP connection issues and provide detailed information.
    
    Args:
        rtsp_url: RTSP stream URL to diagnose
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnosis = {
        'url': rtsp_url,
        'can_open': False,
        'can_read_frame': False,
        'resolution': None,
        'fps': None,
        'codec': None,
        'connection_time': None,
        'frame_read_time': None,
        'error_message': None
    }
    
    try:
        logger.info(f"Diagnosing RTSP connection: {rtsp_url}")
        
        # Test connection timing
        start_time = time.time()
        cap = create_robust_rtsp_connection(rtsp_url, timeout_ms=45000)  # 45 second timeout
        
        if cap.isOpened():
            diagnosis['can_open'] = True
            diagnosis['connection_time'] = time.time() - start_time
            
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            diagnosis['resolution'] = (width, height)
            diagnosis['fps'] = fps
            diagnosis['codec'] = fourcc
            
            # Test frame reading
            frame_start = time.time()
            ret, frame = cap.read()
            
            if ret and frame is not None:
                diagnosis['can_read_frame'] = True
                diagnosis['frame_read_time'] = time.time() - frame_start
                logger.info(f"Successfully read frame: {width}x{height} @ {fps}fps")
            else:
                diagnosis['error_message'] = "Failed to read frame from stream"
                logger.warning("Failed to read frame from RTSP stream")
                
        else:
            diagnosis['error_message'] = "Failed to open RTSP stream"
            logger.error("Failed to open RTSP stream")
            
        cap.release()
        
    except Exception as e:
        diagnosis['error_message'] = str(e)
        logger.error(f"Exception during RTSP diagnosis: {e}")
    
    return diagnosis
