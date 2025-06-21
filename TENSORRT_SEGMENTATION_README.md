# TensorRT Instance Segmentation

This script provides real-time instance segmentation with TensorRT optimization for improved performance.

## Features

- **TensorRT Optimization**: Automatically converts PyTorch models to TensorRT engines for faster inference
- **Real-time Segmentation**: Provides instance segmentation masks with transparency overlay
- **Multi-camera Support**: Supports multiple RTSP cameras and local webcams
- **GPU Acceleration**: Leverages TensorRT for optimal GPU performance
- **Automatic Fallback**: Falls back to PyTorch model if TensorRT export fails

## Usage

### Basic Usage

```bash
python main_tensorrt_segmentation.py
```

### Environment Configuration

Set the TensorRT segmentation model in your `.env` file:

```bash
# TensorRT Segmentation Model Configuration
TENSORRT_SEGMENTATION_MODEL_NAME=yolo11m-seg.engine

# Camera Configuration
CAMERA_STREAMS=0  # or rtsp://user:pass@ip:port/stream

# Performance Settings
MODEL_CONFIDENCE=0.5
MODEL_IOU_THRESHOLD=0.45
```

### Model Requirements

The script requires:
1. **Base segmentation model** (e.g., `yolo11m-seg.pt`) - will be automatically downloaded
2. **TensorRT engine** (e.g., `yolo11m-seg.engine`) - will be automatically generated from the base model

### Automatic TensorRT Export

If the TensorRT engine file doesn't exist, the script will:
1. Load the corresponding PyTorch model (`.pt` file)
2. Export it to TensorRT format (`.engine` file)
3. Save the engine for future use

### Controls

- **'q' or ESC**: Quit the application
- **'s'**: Save screenshots of all camera feeds
- **'r'**: Reset performance statistics

### Performance

TensorRT optimization typically provides:
- 2-5x faster inference compared to PyTorch
- Lower GPU memory usage
- Better throughput for real-time applications

### Troubleshooting

1. **TensorRT not available**: Falls back to PyTorch model automatically
2. **Model not found**: Ensure the base `.pt` model exists in the `models/` directory
3. **Export fails**: Check GPU compatibility and TensorRT installation

### Model Files Structure

```
models/
├── yolo11m-seg.pt      # Base PyTorch model
└── yolo11m-seg.engine  # TensorRT engine (auto-generated)
```
