#!/usr/bin/env python3
"""
Download YOLOv11 segmentation model for instance segmentation.
"""
import os
from ultralytics import YOLO
from loguru import logger

def download_segmentation_model():
    """Download YOLOv11 segmentation model."""
    model_dir = "models"
    model_name = "yolo11n-seg.pt"
    model_path = os.path.join(model_dir, model_name)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        logger.info(f"Segmentation model already exists: {model_path}")
        return model_path
    
    try:
        logger.info(f"Downloading YOLOv11 segmentation model: {model_name}")
        
        # Download the model - this will automatically download it
        model = YOLO(model_name)
        
        # Save it to our models directory
        model.save(model_path)
        
        logger.info(f"Model downloaded successfully: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info("You can manually download the model using:")
        logger.info(f"  python -c \"from ultralytics import YOLO; YOLO('{model_name}').save('{model_path}')\"")
        return None

if __name__ == "__main__":
    download_segmentation_model()
