# 🚀 Real-time Object Detection System - Quick Start Guide

## ✅ Installation Complete!

Your real-time object detection system has been successfully set up and tested. Here's what you can do:

## 🎮 Running the Application

### Option 1: Using Batch File (Recommended)
```cmd
run.bat
```
Double-click `run.bat` or run it from the command line.

### Option 2: Using PowerShell
```powershell
.\run.ps1
```

### Option 3: Direct Python Execution
```cmd
uv run python main.py
```

## 🎬 Testing Without Camera

Run the demo to see object detection in action without needing a camera:
```cmd
uv run python demo.py
```

## 🎯 Application Controls

When running the main application:

- **Q** or **ESC**: Quit the application
- **S**: Save screenshots of all camera feeds
- **R**: Reset performance statistics

## 📷 Camera Configuration

### Default Setup
The system is configured to use:
- Camera 0 (default webcam)

### Adding RTSP Cameras
Edit `config.py` and modify the `DEFAULT_STREAMS` list:

```python
DEFAULT_STREAMS = [
    0,  # Default webcam
    "rtsp://username:password@192.168.1.100:554/stream1",
    "rtsp://admin:password@192.168.1.101:554/cam/realmonitor",
]
```

### Using Environment Variables
Set camera streams via environment variable:
```cmd
set CAMERA_STREAMS=0,rtsp://user:pass@192.168.1.100:554/stream1
```

## ⚙️ Configuration Options

### Model Settings (config.py)
- `MODEL_NAME`: YOLOv11 model file (default: "yolo11n.pt")
- `MODEL_CONFIDENCE`: Detection confidence threshold (0.0-1.0)
- `MODEL_IOU_THRESHOLD`: IoU threshold for NMS

### Performance Settings
- `WINDOW_WIDTH/HEIGHT`: Display window dimensions
- `DISPLAY_FPS`: Show FPS and metrics on screen
- `FRAME_BUFFER_SIZE`: Camera frame buffer size

## 🖥️ System Information

✅ **Python**: 3.12+ (using uv virtual environment)
✅ **PyTorch**: 2.7.1+cpu (CPU version installed)
✅ **OpenCV**: 4.11.0
✅ **YOLOv11**: Model downloaded (80 object classes)
✅ **Dependencies**: All required packages installed

## 📊 Performance Notes

- **GPU**: Currently using CPU version of PyTorch
- **Detection Speed**: ~5.5 seconds per frame (CPU)
- **Model**: YOLOv11 Nano (fastest, good accuracy)

### To Enable GPU Acceleration
1. Install NVIDIA CUDA toolkit
2. Install PyTorch with CUDA support:
   ```cmd
   uv add torch torchvision --index https://download.pytorch.org/whl/cu121
   ```

## 🔧 Troubleshooting

### No Camera Detected
1. Check camera connections
2. Verify camera index (try 0, 1, 2)
3. Test with built-in camera app

### RTSP Connection Issues
1. Test RTSP URL in VLC media player
2. Check network connectivity
3. Verify camera credentials
4. Increase `RTSP_TIMEOUT` in config.py

### Performance Issues
1. Use lighter YOLO model (yolo11n.pt is already the lightest)
2. Reduce camera resolution
3. Close unnecessary applications
4. Consider GPU acceleration

## 📁 Project Files

```
yolo-demo1/
├── main.py              # Main application
├── demo.py              # Camera-free demo
├── config.py            # Configuration settings
├── detector.py          # YOLO detection logic
├── camera_stream.py     # Camera handling
├── utils.py             # Utility functions
├── run.bat              # Windows batch launcher
├── run.ps1              # PowerShell launcher
├── test_system.py       # System tests
├── setup.py             # Setup script
├── README.md            # Full documentation
├── pyproject.toml       # Dependencies
└── yolo11n.pt           # YOLO model file
```

## 🎯 Example RTSP URLs

### Common Camera Brands
```
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.101:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://user:password@192.168.1.102/axis-media/media.amp

# Foscam
rtsp://admin:password@192.168.1.103:88/videoMain
```

## 📈 Next Steps

1. **Test with your cameras**: Run the application and test with your camera setup
2. **Optimize performance**: Adjust settings based on your hardware
3. **Customize detection**: Modify confidence thresholds and detection classes
4. **Add features**: Extend the system with recording, alerts, or analysis

## 🆘 Support

- Check `README.md` for detailed documentation
- Run `uv run python demo.py` to test without cameras
- Review logs for error messages
- Ensure all cables and network connections are secure

---

**🎉 Your object detection system is ready to use!**

Start with: `run.bat` or `uv run python demo.py`
