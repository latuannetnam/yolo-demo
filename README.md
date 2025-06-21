# Real-time Object Detection System

A comprehensive Python application for real-time object detection from multiple camera streams using YOLOv11.

## 🚀 Features

- **Multi-camera Support**: Handle multiple RTSP streams and local cameras simultaneously
- **Real-time Detection**: YOLOv11-powered object detection with GPU acceleration
- **Live Display**: Separate windows for each camera stream with colored bounding boxes
- **Performance Monitoring**: Real-time FPS and detection time display
- **Thread-safe**: Efficient multi-threaded camera stream handling
- **Offline Operation**: Uses pre-trained models that work without internet connection

## 🛠️ Technical Stack

- **Python 3.12+**
- **YOLOv11** (Ultralytics) for object detection
- **OpenCV** for video processing and display
- **PyTorch** for GPU acceleration
- **Multi-threading** for concurrent camera handling

## 📋 Requirements

- Windows 11 with NVIDIA GPU (recommended)
- Python 3.12 or higher
- NVIDIA CUDA toolkit (for GPU acceleration)
- Webcam or RTSP camera streams

## 🔧 Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd d:\latuan\Programming\object-detection\yolo-demo1
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   uv run python main.py
   ```

## ⚙️ Configuration

### Camera Streams

Edit `config.py` to configure your camera streams:

```python
DEFAULT_STREAMS = [
    0,  # Default webcam
    "rtsp://username:password@192.168.1.100:554/stream1",  # RTSP camera
    "rtsp://admin:123456@192.168.1.101:554/cam/realmonitor",  # Another RTSP camera
]
```

### Environment Variables

You can also set camera streams via environment variable:
```bash
$env:CAMERA_STREAMS = "0,rtsp://user:pass@192.168.1.100:554/stream1"
```

### GPU Configuration

GPU usage is automatically detected. To force CPU usage, modify `config.py`:
```python
USE_GPU = False
```

## 🎮 Usage

### Basic Usage

```bash
uv run python main.py
```

### Controls

- **Q** or **ESC**: Quit the application
- **S**: Save screenshots of all camera feeds
- **R**: Reset performance statistics

### Command Line Options

The system automatically detects available cameras and starts processing. Each camera stream appears in a separate window with:

- Real-time object detection results
- FPS counter
- Detection time metrics
- Object count per frame

## 📁 Project Structure

```
yolo-demo1/
├── main.py              # Main application entry point
├── config.py            # Configuration settings
├── detector.py          # YOLOv11 object detection logic
├── camera_stream.py     # Multi-camera stream handling
├── pyproject.toml       # Project dependencies
├── README.md           # This file
└── instruction.md      # Original project requirements
```

## 🔧 Configuration Options

### Model Settings
- `MODEL_NAME`: YOLOv11 model file (default: "yolo11n.pt")
- `MODEL_CONFIDENCE`: Detection confidence threshold (0.0-1.0)
- `MODEL_IOU_THRESHOLD`: IoU threshold for NMS

### Display Settings
- `WINDOW_WIDTH/HEIGHT`: Display window dimensions
- `DISPLAY_FPS`: Show FPS and metrics on screen
- `BBOX_COLORS`: Colors for bounding boxes per camera

### Performance Settings
- `FRAME_BUFFER_SIZE`: Camera frame buffer size
- `RTSP_TIMEOUT`: RTSP connection timeout

## 🚨 Troubleshooting

### Common Issues

1. **No cameras detected**:
   - Check camera connections
   - Verify RTSP URLs and credentials
   - Test with camera index 0 for built-in webcam

2. **GPU not detected**:
   - Install NVIDIA CUDA toolkit
   - Update GPU drivers
   - Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Poor performance**:
   - Reduce camera resolution
   - Use lighter YOLO model (yolo11n.pt)
   - Increase frame buffer size
   - Close unnecessary applications

4. **RTSP connection issues**:
   - Check network connectivity
   - Verify camera IP and port
   - Test RTSP URL in VLC media player
   - Increase RTSP timeout value

### Performance Optimization

- **GPU Memory**: Monitor GPU memory usage with `nvidia-smi`
- **CPU Usage**: Use lighter YOLO models for CPU-only setups
- **Network**: Ensure stable network for RTSP streams
- **Threading**: Adjust frame buffer size based on system capabilities

## 📊 Performance Metrics

The application displays real-time metrics:

- **FPS**: Frames per second per camera
- **Detection Time**: Time taken for object detection (ms)
- **Object Count**: Number of detected objects per frame
- **System Stats**: Overall processing statistics

## 🤝 Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Update documentation for new features
4. Test with multiple camera types

## 📄 License

This project is for educational and development purposes. Please respect the licenses of the underlying libraries (Ultralytics, OpenCV, etc.).

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv11
- **OpenCV** community
- **PyTorch** team
- **Python** ecosystem

---

For questions or issues, please check the troubleshooting section or create an issue in the project repository.