# YOLO11 Heatmap Object Detection System

This document describes the heatmap visualization functionality built on top of the existing object detection system, implementing advanced data visualization using Ultralytics YOLO11.

## 🔥 Overview

The heatmap detection system extends the base object detection capabilities with dynamic visualization that shows object movement patterns over time. It uses color-coded intensity maps to represent areas of high and low activity, making it easy to identify patterns, trends, and anomalies in object behavior.

## 🚀 Features

### Core Functionality
- **Real-time Heatmap Generation**: Dynamic heatmap overlay on live video streams
- **Multi-Camera Support**: Independent heatmap tracking for each camera source
- **Object Tracking Integration**: Built-in object tracking for accurate movement pattern analysis
- **Multiple Colormap Options**: 10+ different visualization styles to choose from

### Interactive Controls
- **Live Colormap Switching**: Change visualization style in real-time with 'c' key
- **Screenshot Capture**: Save heatmap visualizations with 's' key
- **Statistics Monitoring**: Real-time performance metrics display
- **Help System**: Built-in help accessible with 'h' key

### Advanced Features
- **Configurable Object Classes**: Filter specific object types for heatmap generation
- **Confidence Thresholding**: Adjustable detection confidence levels
- **GPU Acceleration**: CUDA support for enhanced performance
- **Logging System**: Comprehensive logging with rotation and retention

## 🎨 Available Colormaps

| Colormap | Description | Best For |
|----------|-------------|----------|
| JET | Classic blue-to-red gradient | General purpose, high contrast |
| HOT | Black-red-yellow-white heat colors | Temperature-like visualization |
| VIRIDIS | Perceptually uniform purple-to-yellow | Scientific visualization, accessibility |
| PLASMA | Purple-to-pink-to-yellow | Modern, vibrant visualization |
| INFERNO | Black-to-red-to-yellow | Dark backgrounds, dramatic effect |
| MAGMA | Black-to-purple-to-white | Elegant, sophisticated look |
| TURBO | Google Turbo colormap | High dynamic range, smooth gradients |
| RAINBOW | Full spectrum rainbow | Maximum color differentiation |
| OCEAN | Ocean-inspired blue-green | Calming, water-themed visualization |
| COOL | Cool cyan-to-magenta | Professional, modern appearance |

## 📊 Use Cases

### Retail Analytics
- **Customer Flow Analysis**: Track shopping patterns and popular areas
- **Product Placement Optimization**: Identify high-traffic zones for strategic placement
- **Queue Management**: Analyze waiting patterns and optimize service points

### Traffic & Transportation
- **Vehicle Flow Monitoring**: Visualize traffic patterns on roads and intersections
- **Parking Utilization**: Track parking space usage over time
- **Public Transport Analysis**: Monitor passenger flow at stations and stops

### Security & Surveillance
- **Intrusion Detection**: Identify unusual movement patterns in restricted areas
- **Perimeter Monitoring**: Track activity around building perimeters
- **Crowd Management**: Monitor large gatherings and events

### Smart Buildings
- **Space Utilization**: Analyze how different areas of buildings are used
- **Energy Optimization**: Correlate movement patterns with HVAC usage
- **Emergency Planning**: Understand evacuation routes and bottlenecks

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Install required packages
pip install ultralytics opencv-python loguru python-dotenv numpy
```

### Configuration
The system uses the same configuration as the base object detection system. Key settings:

```python
# Model Configuration
MODEL_NAME = "yolo11n.pt"  # Or any YOLO11 model
MODEL_CONFIDENCE = 0.5     # Detection confidence threshold
MODEL_IOU_THRESHOLD = 0.45 # IoU threshold for tracking

# Display Settings
WINDOW_WIDTH = 640         # Display window width
WINDOW_HEIGHT = 480        # Display window height
DISPLAY_FPS = True         # Show FPS information
```

## 🚀 Usage

### Basic Usage
```bash
# Run heatmap detection with default settings
python main_heatmap.py
```

### Demo Mode
```bash
# Run interactive demo with explanations
python demo_heatmap.py
```

### Advanced Configuration
```python
from main_heatmap import HeatmapDetectionSystem

# Create system with custom configuration
system = HeatmapDetectionSystem()

# Customize heatmap settings
system.heatmap_config['colormap'] = cv2.COLORMAP_VIRIDIS
system.heatmap_config['classes'] = [0, 2, 3]  # Only persons, cars, motorbikes
system.heatmap_config['conf'] = 0.7           # Higher confidence threshold

# Run the system
system.run()
```

## ⌨️ Interactive Controls

| Key | Action | Description |
|-----|--------|-------------|
| `q` / `ESC` | Quit | Exit the application |
| `s` | Screenshot | Save current frame with heatmap |
| `r` | Reset Stats | Reset performance statistics |
| `c` | Cycle Colormap | Switch to next colormap |
| `h` | Help | Display help information |

## 📈 Performance Monitoring

The system provides real-time performance metrics:

- **FPS**: Frames per second for each camera
- **Processing Time**: Object detection and heatmap generation time
- **Display Time**: Frame rendering and display time
- **Total Frames**: Cumulative frame count

## 🔧 Technical Implementation

### Architecture
```
main_heatmap.py
├── HeatmapDetectionSystem (Main orchestrator)
├── MultiCameraManager (Camera stream handling)
├── Ultralytics Heatmap (Core heatmap generation)
└── Config (System configuration)
```

### Key Components

#### HeatmapDetectionSystem
- Main system orchestrator
- Manages multiple camera streams
- Handles user interactions and controls
- Provides performance monitoring

#### Heatmap Integration
- Uses Ultralytics solutions.Heatmap
- Automatic object tracking
- Dynamic intensity accumulation
- Configurable visualization options

#### Multi-Camera Support
- Independent heatmap for each camera
- Synchronized processing across streams
- Individual window management
- Per-camera statistics

## 🎯 Optimization Tips

### Performance
1. **GPU Utilization**: Ensure CUDA is available for maximum performance
2. **Model Selection**: Use appropriate model size (nano/small for speed, large/xlarge for accuracy)
3. **Resolution**: Balance detection accuracy with processing speed
4. **Frame Rate**: Consider camera frame rate vs processing capability

### Visualization
1. **Colormap Selection**: Choose colormaps appropriate for your use case
2. **Confidence Thresholds**: Adjust to reduce noise while maintaining detection accuracy
3. **Class Filtering**: Focus on relevant object classes for cleaner heatmaps
4. **Accumulation Time**: Allow sufficient time for meaningful pattern development

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **Camera Connection Issues**: Automatic retry and graceful degradation
- **Model Loading Failures**: Clear error messages and fallback options
- **GPU Unavailability**: Automatic fallback to CPU processing
- **Memory Management**: Efficient resource usage and cleanup

## 📝 Logging

Comprehensive logging system with:
- **File Rotation**: Automatic log file rotation at 10MB
- **Retention Policy**: 7-day log retention
- **Multiple Levels**: INFO, WARNING, ERROR logging levels
- **Structured Output**: Clear, searchable log format

## 🔮 Future Enhancements

Planned improvements include:
- **Region-based Heatmaps**: Focus on specific areas of interest
- **Temporal Analysis**: Time-based heatmap comparisons
- **Export Capabilities**: Save heatmap data for external analysis
- **Web Interface**: Browser-based monitoring and control
- **Alert System**: Notifications for unusual patterns

## 📚 References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [Heatmap Visualization Guide](https://docs.ultralytics.com/guides/heatmaps/)
- [OpenCV Colormap Documentation](https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html)

---

For technical support or feature requests, please refer to the main project documentation or create an issue in the project repository.
