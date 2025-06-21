# Optimized Real-time Object Detection System

This is an advanced, performance-optimized version of the real-time object detection system with comprehensive improvements for better throughput, lower latency, and enhanced resource utilization.

## 🚀 Performance Improvements

### Major Optimizations Implemented

1. **Parallel Processing Architecture**
   - Multi-threaded detection pipeline
   - Asynchronous frame processing
   - Background detection workers

2. **Smart Frame Management**
   - Intelligent frame skipping when system is overloaded
   - Adaptive frame rate control
   - Queue-based buffering system

3. **Memory Optimization**
   - Frame buffer pooling and reuse
   - Pre-allocated resize buffers
   - Reduced memory allocations

4. **GPU Batch Processing**
   - Batch inference for better GPU utilization
   - Dynamic batch size optimization
   - Fallback to individual processing

5. **Detection Caching**
   - Short-term detection result caching
   - Reduced redundant computations
   - Configurable cache timeout

6. **Non-blocking Display**
   - Separate display thread
   - Throttled display updates
   - Smooth UI responsiveness

## 📊 Expected Performance Gains

- **3-5x increase** in overall FPS
- **50-70% reduction** in frame-to-display latency
- **Better GPU utilization** with batch processing
- **Reduced memory** allocations and garbage collection
- **Smoother display** updates with non-blocking UI

## 🛠 Installation

### Prerequisites

```bash
# Install required packages
pip install ultralytics opencv-python loguru python-dotenv psutil

# Optional: For performance monitoring
pip install psutil
```

### Quick Start

1. **Run the optimized system:**
   ```bash
   python run_optimized.py
   ```

2. **Run performance comparison:**
   ```bash
   python performance_comparison.py --duration 30
   ```

3. **Test only optimized version:**
   ```bash
   python performance_comparison.py --optimized-only --duration 60
   ```

## ⚙️ Configuration

The optimized system uses `config_optimized.py` with additional performance parameters:

### Performance Settings

```python
# Threading settings
DETECTION_THREADS = 2          # Number of detection worker threads

# Queue settings
FRAME_QUEUE_SIZE = 100         # Main frame processing queue
RESULT_QUEUE_SIZE = 50         # Detection results queue
DISPLAY_QUEUE_SIZE = 30        # Display queue

# Frame management
MIN_FRAME_INTERVAL = 0.033     # ~30 FPS (1/30 seconds)
MAX_DETECTION_TIME = 0.1       # 100ms max detection time
FRAME_SKIP_THRESHOLD = 5       # Skip every 5th frame if behind

# Batch processing
BATCH_SIZE = 4                 # Frames to process in batch
BATCH_TIMEOUT = 0.05           # 50ms max wait for batch

# Memory optimization
USE_FRAME_POOLING = True       # Enable frame buffer pooling
FRAME_POOL_SIZE = 20           # Max pooled frames

# Cache settings
DETECTION_CACHE_SIZE = 100     # Max cached detections
DETECTION_CACHE_TIMEOUT = 0.1  # 100ms cache timeout

# Display optimization
MAX_DISPLAY_FPS = 60           # Maximum display refresh rate
DISPLAY_THROTTLE = 0.0166      # 1/60 for 60 FPS
```

### Environment Variables

Create a `.env` file to override default settings:

```env
# Model settings
MODEL_NAME=yolo11n.pt
MODEL_CONFIDENCE=0.5
MODEL_IOU_THRESHOLD=0.45

# Performance settings
DETECTION_THREADS=2
BATCH_SIZE=4
FRAME_QUEUE_SIZE=100
MIN_FRAME_INTERVAL=0.033

# Camera settings
CAMERA_STREAMS=0,1
# CAMERA_STREAMS=rtsp://admin:password@192.168.1.100:554/stream,0

# GPU settings
USE_GPU=true
DEVICE=cuda
```

## 🎮 Controls

### Keyboard Shortcuts

- **`q` or `ESC`**: Quit the application
- **`s`**: Save screenshots of all cameras
- **`r`**: Reset performance statistics
- **`p`**: Print current performance statistics

## 📈 Performance Monitoring

The optimized system includes comprehensive performance monitoring:

### Real-time Statistics

- Total frames processed
- Frames processed vs skipped
- Average detection time
- Queue overflow counts
- Batch detection statistics
- FPS per camera

### Performance Metrics

```python
# Access performance stats programmatically
system = OptimizedObjectDetectionSystem()
stats = system.performance_monitor.stats

print(f"Processed: {stats['processed_frames']}")
print(f"Skipped: {stats['skipped_frames']}")
print(f"Avg detection time: {stats['detection_time']:.2f}ms")
```

## 🔧 Advanced Configuration

### Automatic Optimization

The system automatically optimizes settings based on hardware:

```python
# Automatic thread count based on CPU cores
optimal_threads = Config.get_optimal_thread_count()

# Automatic batch size based on GPU memory
optimal_batch = Config.get_optimized_batch_size()

# Print current configuration
Config.print_config_summary()
```

### Custom Optimization

For specific hardware configurations:

```python
# High-end GPU setup
BATCH_SIZE = 8
DETECTION_THREADS = 4
FRAME_QUEUE_SIZE = 200

# Memory-constrained setup
BATCH_SIZE = 1
FRAME_POOL_SIZE = 10
DETECTION_CACHE_SIZE = 50

# High-throughput setup
MIN_FRAME_INTERVAL = 0.02  # 50 FPS
DISPLAY_THROTTLE = 0.033   # 30 FPS display
```

## 🐛 Troubleshooting

### Common Issues

1. **High CPU Usage**
   ```env
   DETECTION_THREADS=1
   BATCH_SIZE=2
   ```

2. **Memory Issues**
   ```env
   FRAME_POOL_SIZE=5
   DETECTION_CACHE_SIZE=20
   FRAME_QUEUE_SIZE=50
   ```

3. **GPU Memory Issues**
   ```env
   BATCH_SIZE=1
   USE_GPU=false
   ```

### Performance Tuning

1. **Monitor system resources:**
   ```bash
   # Run with performance monitoring
   python performance_comparison.py --optimized-only --duration 120
   ```

2. **Adjust based on hardware:**
   - **Low-end systems**: Reduce batch size and thread count
   - **High-end systems**: Increase queue sizes and batch processing
   - **Memory-limited**: Enable frame pooling and reduce cache sizes

## 📋 System Requirements

### Minimum Requirements

- Python 3.8+
- 4GB RAM
- OpenCV 4.5+
- YOLOv11 compatible GPU (optional)

### Recommended for Optimal Performance

- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- Multi-core CPU (4+ cores)

## 🔍 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frame         │    │   Detection     │    │   Display       │
│   Collection    │───▶│   Workers       │───▶│   Thread        │
│   Loop          │    │   (Parallel)    │    │   (Non-blocking)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frame Queue   │    │   Result Queue  │    │   UI Updates    │
│   (Buffered)    │    │   (Processed)   │    │   (Throttled)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📄 License

Same license as the original project.

## 🤝 Contributing

1. Test performance improvements
2. Report issues with specific hardware configurations
3. Suggest additional optimizations
4. Benchmark different model types

## 📞 Support

For performance-related issues or optimization questions, please include:

- Hardware specifications (CPU, GPU, RAM)
- Current configuration settings
- Performance monitoring output
- Specific use case requirements
