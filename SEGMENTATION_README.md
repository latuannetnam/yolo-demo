# Instance Segmentation with YOLOv11

This document describes how to use the instance segmentation feature with YOLOv11.

## Overview

The `main_segmentation.py` file implements real-time instance segmentation using YOLOv11 segmentation models. It provides the same multi-camera support as the object detection system but with pixel-level segmentation masks.

## Features

- **Real-time Instance Segmentation**: Uses YOLOv11 segmentation models for pixel-perfect object detection
- **Multi-camera Support**: Works with RTSP streams and local cameras
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Segmentation Masks**: Overlays colored masks on detected objects with adjustable transparency
- **Performance Monitoring**: Real-time FPS and processing time display

## Setup

### 1. Environment Variable

Add the segmentation model name to your `.env` file:

```bash
# Segmentation model settings
SEGMENTATION_MODEL_NAME=yolo11n-seg.pt
```

The system will use the same `MODEL_DIR` variable for the segmentation model path.

### 2. Download Segmentation Model

Run the download script to get the YOLOv11 segmentation model:

```bash
python download_segmentation_model.py
```

Or manually download using:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt').save('models/yolo11n-seg.pt')"
```

### 3. Available Segmentation Models

YOLOv11 offers several segmentation model variants:

- `yolo11n-seg.pt` - Nano (fastest, least accurate)
- `yolo11s-seg.pt` - Small
- `yolo11m-seg.pt` - Medium  
- `yolo11l-seg.pt` - Large
- `yolo11x-seg.pt` - Extra Large (slowest, most accurate)

## Usage

### Running the Segmentation System

```bash
python main_segmentation.py
```

### Controls

- **'q' or ESC**: Quit the application
- **'s'**: Save screenshots of current frames with segmentations
- **'r'**: Reset performance statistics

## Configuration

The segmentation system uses the same configuration as the object detection system:

- **Camera Settings**: Use `CAMERA_STREAMS` environment variable
- **Model Confidence**: `MODEL_CONFIDENCE` (default: 0.5)
- **IoU Threshold**: `MODEL_IOU_THRESHOLD` (default: 0.45)
- **Display Settings**: `WINDOW_WIDTH`, `WINDOW_HEIGHT`, `DISPLAY_FPS`

## Output

The system provides:

1. **Segmentation Masks**: Colored overlays showing the exact shape of detected objects
2. **Bounding Boxes**: Traditional rectangular bounds around objects
3. **Class Labels**: Object class name and confidence score
4. **Performance Metrics**: FPS and processing time per frame

## Differences from Object Detection

| Feature | Object Detection | Instance Segmentation |
|---------|------------------|----------------------|
| Output | Bounding boxes only | Bounding boxes + pixel masks |
| Model | `yolo11n.pt` | `yolo11n-seg.pt` |
| Processing | Faster | Slower (more detailed) |
| Memory Usage | Lower | Higher |
| Accuracy | Box-level | Pixel-level |

## Troubleshooting

### Model Not Found
If you get a "model not found" error:
1. Run `python download_segmentation_model.py`
2. Check that the model file exists in the `models/` directory
3. Verify the `SEGMENTATION_MODEL_NAME` environment variable

### Low Performance
- Try a smaller model variant (e.g., `yolo11n-seg.pt`)
- Ensure GPU acceleration is working
- Reduce input resolution if needed

### Memory Issues
- Use CPU instead of GPU by setting `USE_GPU=false`
- Try a smaller model variant
- Reduce the number of concurrent camera streams

## Example Output

The segmentation system will display:
- Camera feed with colored segmentation masks overlaid on detected objects
- Bounding boxes around each object
- Class labels with confidence scores
- Real-time FPS and processing time statistics
