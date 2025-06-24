"""
Configuration settings for the real-time object detection system.

ZONE CONFIGURATION EXAMPLES:
============================

1. Simple rectangular zone:
   ZONE_IN_POLYGONS='[[[100, 50], [400, 50], [400, 300], [100, 300]]]'

2. Multiple zones:
   ZONE_IN_POLYGONS='[
       [[806, 121], [1390, 121], [1390, 77], [806, 77]],
       [[690, 300], [1530, 300], [1530, 230], [690, 230]]
   ]'

3. Complex polygon (5+ points):
   ZONE_IN_POLYGONS='[[[100, 100], [200, 50], [300, 100], [250, 200], [150, 200]]]'

COORDINATE SYSTEM:
==================
- Origin (0, 0) is at top-left corner of the image
- X increases from left to right
- Y increases from top to bottom
- All coordinates should be within image dimensions

ZONE TYPES:
===========
- ZONE_IN_POLYGONS: Entry detection zones (objects entering these areas)
- ZONE_OUT_POLYGONS: Exit detection zones (objects leaving these areas)

LINE ZONE CONFIGURATION EXAMPLES:
================================

1. Single horizontal line:
   LINE_ZONES='[[[0, 540], [1920, 540]]]'

2. Multiple lines:
   LINE_ZONES='[[[0, 540], [1920, 540]], [[0, 800], [1920, 800]]]'"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Any
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv(override=True)

class Config:
    """Configuration class for object detection system."""
      # Model settings
    MODEL_DIR = os.getenv("MODEL_DIR", "models")  # Directory containing model files
    MODEL_NAME = os.getenv("MODEL_NAME", "yolo11n.pt")  # YOLOv11 nano model (fastest)    
    SEGMENTATION_MODEL_NAME = os.getenv("SEGMENTATION_MODEL_NAME", "yolo11n-seg.pt")  # YOLOv11 segmentation model
    TENSORRT_SEGMENTATION_MODEL_NAME = os.getenv("TENSORRT_SEGMENTATION_MODEL_NAME", "yolo11m-seg.engine")  # YOLOv11 segmentation model for TensorRT optimization
    PROMPT_MODEL_NAME = os.getenv("PROMPT_MODEL_NAME", "yoloe-11s-seg.pt")  # YOLOE prompt-based model
    TENSORRT_PROMPT_MODEL_NAME = os.getenv("TENSORRT_PROMPT_MODEL_NAME", "yoloe-11m-seg.engine")  # YOLOE model for TensorRT prompt detection
    INITIAL_PROMPTS = os.getenv("INITIAL_PROMPTS", "car")  # Default prompts for detection
    MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.5"))
    MODEL_IOU_THRESHOLD = float(os.getenv("MODEL_IOU_THRESHOLD", "0.45"))
    SLICE_WORKERS = int(os.getenv("SLICE_WORKERS", "1"))
    NUM_TILES = int(os.getenv("NUM_TILES", "4"))

    # Feature toggles
    USE_SLICER = os.getenv("USE_SLICER", "true").lower() == "true"
    USE_SMOOTHER = os.getenv("USE_SMOOTHER", "true").lower() == "true"
    USE_LINE_ZONE = os.getenv("USE_LINE_ZONE", "true").lower() == "true"
    USE_POLYGON_ZONE = os.getenv("USE_POLYGON_ZONE", "true").lower() == "true"
    USE_HEATMAP = os.getenv("USE_HEATMAP", "true").lower() == "true"
    USE_TRACE = os.getenv("USE_TRACE", "true").lower() == "true"

    LINE_ZONES_STR = os.getenv("LINE_ZONES", "[]")
    LINE_ZONES: List[np.ndarray] = []
    
    # GPU settings
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    DEVICE = os.getenv("DEVICE", "cuda" if USE_GPU else "cpu")
    
    # Display settings
    WINDOW_WIDTH = int(os.getenv("WINDOW_WIDTH", "640"))
    WINDOW_HEIGHT = int(os.getenv("WINDOW_HEIGHT", "480"))
    DISPLAY_FPS = os.getenv("DISPLAY_FPS", "true").lower() == "true"
    
    # Camera settings
    RTSP_TIMEOUT = int(os.getenv("RTSP_TIMEOUT", "30"))  # seconds
    RTSP_RECONNECT_TIMEOUT = int(os.getenv("RTSP_RECONNECT_TIMEOUT", "45"))  # seconds for reconnection
    FRAME_BUFFER_SIZE = int(os.getenv("FRAME_BUFFER_SIZE", "1"))
    
    # RTSP-specific settings
    RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")  # tcp or udp
    RTSP_STIMEOUT = int(os.getenv("RTSP_STIMEOUT", "5000000"))  # microseconds (5 seconds)
    MAX_RECONNECT_ATTEMPTS = int(os.getenv("MAX_RECONNECT_ATTEMPTS", "3"))
      # Color settings for bounding boxes (BGR format)
    BBOX_COLORS = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
      # Default camera streams (can be RTSP URLs or camera indices)
    DEFAULT_STREAMS = [
        "0",  # Default webcam
        # Add RTSP URLs here, e.g.:
        # "rtsp://admin:admin123@117.0.0.18:5551/cam/realmonitor?channel=1&subtype=0",
        # "rtsp://admin:password@192.168.1.100:554/stream1",
    ]
    
    # YouTube settings
    YOUTUBE_URL = os.getenv("YOUTUBE_URL", "https://www.youtube.com/watch?v=zpreb8_1qA4")  # YouTube video URL
    YOUTUBE_QUALITY = os.getenv("YOUTUBE_QUALITY", "720p")  # Video quality preference
    YOUTUBE_STREAM_TIMEOUT = int(os.getenv("YOUTUBE_STREAM_TIMEOUT", "30"))  # seconds
    PLAYBACK_FPS = int(os.getenv("PLAYBACK_FPS", "30")) # Playback FPS

    @classmethod
    def get_line_zones(cls) -> List[np.ndarray]:
        """Parse LINE_ZONES from environment variable."""
        try:
            zones = json.loads(cls.LINE_ZONES_STR)
            if not isinstance(zones, list):
                logger.warning(f"LINE_ZONES is not a list, but {type(zones)}. No lines will be used.")
                return []
            return [np.array(zone, dtype=np.int32) for zone in zones]
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON for LINE_ZONES: {cls.LINE_ZONES_STR}. No lines will be used.")
            return []

    @classmethod
    def get_full_model_name(cls) -> str:
        """Get the full path to the model file by combining MODEL_DIR and MODEL_NAME."""
        return os.path.join(cls.MODEL_DIR, cls.MODEL_NAME)    
    
    @classmethod
    def get_full_segmentation_model_name(cls) -> str:
        """Get the full path to the segmentation model file by combining MODEL_DIR and SEGMENTATION_MODEL_NAME."""
        return os.path.join(cls.MODEL_DIR, cls.SEGMENTATION_MODEL_NAME)
    
    @classmethod
    def get_full_tensorrt_segmentation_model_name(cls) -> str:
        """Get the full path to the TensorRT segmentation model file by combining MODEL_DIR and TENSORRT_SEGMENTATION_MODEL_NAME."""
        return os.path.join(cls.MODEL_DIR, cls.TENSORRT_SEGMENTATION_MODEL_NAME)
    
    @classmethod
    def get_full_prompt_model_name(cls) -> str:
        """Get the full path to the prompt model file by combining MODEL_DIR and PROMPT_MODEL_NAME."""
        return os.path.join(cls.MODEL_DIR, cls.PROMPT_MODEL_NAME)
    
    @classmethod
    def get_full_tensorrt_prompt_model_name(cls) -> str:
        """Get the full path to the TensorRT prompt model file by combining MODEL_DIR and TENSORRT_PROMPT_MODEL_NAME."""
        return os.path.join(cls.MODEL_DIR, cls.TENSORRT_PROMPT_MODEL_NAME)
    
    @classmethod
    def get_initial_prompts(cls) -> List[str]:
        """Get initial prompts from environment variable."""
        prompts_str = cls.INITIAL_PROMPTS
        if prompts_str:
            return [p.strip() for p in prompts_str.split(',') if p.strip()]
        return ["person", "car"]  # Fallback default
    
    @classmethod
    def get_stream_config(cls) -> List[str]:
        """Get camera stream configuration from environment or defaults."""
        streams_env = os.getenv("CAMERA_STREAMS")
        if streams_env:
            return streams_env.split(",")
        return cls.DEFAULT_STREAMS
    
    @classmethod
    def get_validated_streams(cls) -> List[str]:
        """Get validated camera streams, falling back to demo mode if needed."""
        streams = cls.get_stream_config()
        
        # If no streams configured or streams are invalid, use demo mode
        if not streams or streams == [""]:
            logger.warning("No camera streams configured, using demo mode")
            return ["demo"]
        
        # Check if we should use demo mode
        if "demo" in streams:
            return ["demo"]
          # For now, return configured streams (validation can be added here)
        return streams
    
    @classmethod
    def validate_gpu_availability(cls) -> bool:
        """Check if GPU is available for inference."""
        try:
            import torch
            return torch.cuda.is_available() and cls.USE_GPU
        except ImportError:
            return False
    
    @classmethod
    def get_zone_in_polygons(cls) -> List[np.ndarray]:
        """
        Get entry zone polygons from environment variable or defaults.
        
        Entry zones define areas where objects are monitored for entry detection.
        Each zone is a polygon defined by a list of [x, y] coordinates.
        
        Environment variable format (JSON array of polygons):
        ZONE_IN_POLYGONS='[
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        ]'
        
        Example - Two rectangular zones:
        ZONE_IN_POLYGONS='[
            [[806, 121], [1390, 121], [1390, 77], [806, 77]],
            [[690, 300], [1530, 300], [1530, 230], [690, 230]]
        ]'
        
        Coordinate format:
        - [x, y] where x=horizontal position, y=vertical position
        - Top-left corner of image is [0, 0]
        - Polygon should be closed (first and last points connect automatically)
        - Minimum 3 points required, 4+ recommended for rectangles
        
        Returns:
            List of numpy arrays representing polygon coordinates for entry zones
        """
        zones_env = os.getenv("ZONE_IN_POLYGONS")
        
        if zones_env:
            try:
                # Parse JSON string from environment variable
                zones_data = json.loads(zones_env)
                zones = []
                
                for zone_coords in zones_data:
                    if cls.validate_zone_coordinates(zone_coords):
                        zones.append(np.array(zone_coords, dtype=np.int32))
                
                if zones:
                    logger.info(f"Loaded {len(zones)} entry zones from environment variable.")
                    return zones
                else:
                    logger.warning("No valid zones found in ZONE_IN_POLYGONS.")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ZONE_IN_POLYGONS from environment: {e}")
                logger.info("Using default zone configuration")
            except Exception as e:
                logger.error(f"Error processing ZONE_IN_POLYGONS: {e}")
        
        # Default zones if environment variable not set or invalid
        default_zones = [
            np.array([[806, 121], [1390, 121], [1390, 77], [806, 77]], dtype=np.int32),
            np.array([[690, 300], [1530, 300], [1530, 230], [690, 230]], dtype=np.int32),
            np.array([[271, 681], [520, 681], [520, 450], [271, 450]], dtype=np.int32),
        ]
        
        logger.info(f"Using {len(default_zones)} default zones")
        return default_zones
    
    @classmethod
    def set_zone_in_polygons_env(cls, zones: List[List[List[int]]]) -> str:
        """
        Helper method to set ZONE_IN_POLYGONS environment variable.
        
        Args:
            zones: List of zones, where each zone is a list of [x, y] coordinate pairs
            
        Returns:
            JSON string that can be used as environment variable value
        """
        try:
            zones_json = json.dumps(zones)
            logger.info(f"Generated ZONE_IN_POLYGONS env value: {zones_json}")
            return zones_json
        except Exception as e:
            logger.error(f"Failed to generate zones JSON: {e}")
            return ""
    
    @classmethod
    def get_zone_out_polygons(cls) -> List[np.ndarray]:
        """
        Get exit zone polygons from environment variable or empty list.
        
        Environment variable format (JSON string):
        ZONE_OUT_POLYGONS='[[[950, 282], [1250, 282], [1250, 82], [950, 82]]]
        
        Returns:
            List of numpy arrays representing polygon coordinates
        """
        zones_env = os.getenv("ZONE_OUT_POLYGONS")
        
        if zones_env:
            try:
                zones_data = json.loads(zones_env)
                zones = []
                for zone_coords in zones_data:
                    if cls.validate_zone_coordinates(zone_coords):
                        zones.append(np.array(zone_coords, dtype=np.int32))
                
                if zones:
                    logger.info(f"Loaded {len(zones)} exit zones from environment variable.")
                    return zones
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse ZONE_OUT_POLYGONS: {e}")
            except Exception as e:
                logger.error(f"Error processing ZONE_OUT_POLYGONS: {e}")
        
        # Return empty list for exit zones (entry-only mode)
        logger.info("No exit zones configured - running in entry-only mode")
        return []
    
    @classmethod
    def create_rectangular_zone(cls, x1: int, y1: int, x2: int, y2: int) -> List[List[int]]:
        """
        Helper method to create a rectangular zone from two corner points.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            
        Returns:
            List of [x, y] coordinates forming a rectangle
            
        Example:
            zone = Config.create_rectangular_zone(100, 50, 400, 300)
            # Creates: [[100, 50], [400, 50], [400, 300], [100, 300]]
        """
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    
    @classmethod
    def create_zones_json(cls, zones: List[List[List[int]]]) -> str:
        """
        Helper method to create properly formatted JSON for ZONE_IN_POLYGONS environment variable.
        
        Args:
            zones: List of zones, where each zone is a list of [x, y] coordinate pairs
            
        Returns:
            Formatted JSON string ready for environment variable
            
        Example:
            # Create two rectangular zones
            zone1 = Config.create_rectangular_zone(806, 77, 1390, 121)
            zone2 = Config.create_rectangular_zone(690, 230, 1530, 300)
            zones_json = Config.create_zones_json([zone1, zone2])
            # Set in .env file: ZONE_IN_POLYGONS='[zones_json value]'
        """
        try:
            import json
            zones_json = json.dumps(zones, indent=2)
            logger.info(f"Generated zones JSON:\n{zones_json}")
            return zones_json
        except Exception as e:
            logger.error(f"Failed to generate zones JSON: {e}")
            return "[]"
    
    @classmethod
    def validate_zone_coordinates(cls, zone: List[List[int]]) -> bool:
        """
        Validate zone coordinates to ensure they form a valid polygon.
        
        Args:
            zone: List of [x, y] coordinate pairs
            
        Returns:
            True if zone is valid, False otherwise
        """
        if not isinstance(zone, list) or len(zone) < 3:
            logger.warning(f"Zone must have at least 3 points, got {len(zone) if isinstance(zone, list) else 0}")
            return False
        
        for point in zone:
            if not isinstance(point, list) or len(point) != 2:
                logger.warning(f"Invalid point format in zone: {point}. Each point must be a list of 2 coordinates.")
                return False
            
            if not all(isinstance(coord, (int, float)) and coord >= 0 for coord in point):
                logger.warning(f"Invalid coordinates in point: {point}. Coordinates must be non-negative numbers.")
                return False
    
        return True

    @staticmethod
    def get_selected_class_names() -> List[str]:
        """
        Returns a list of selected class names from the environment variable SELECTED_CLASS_NAMES.
        The environment variable should contain a comma-separated string of class names.
        """
        class_names_str = os.getenv("SELECTED_CLASS_NAMES")
        if not class_names_str:
            return []
        return [name.strip() for name in class_names_str.split(',') if name.strip()]

Config.LINE_ZONES = Config.get_line_zones()
