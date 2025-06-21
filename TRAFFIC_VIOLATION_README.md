# Traffic Light Violation Detection System

## Overview

This system extends the basic object detection capabilities to specifically detect and monitor traffic light violations. Based on the analysis of your traffic intersection image, the system can:

1. **Detect Traffic Lights**: Identify traffic lights in the video feed and determine their current state (Red/Yellow/Green)
2. **Monitor Vehicle Movement**: Track cars, trucks, motorcycles, buses, and bicycles as they move through the intersection
3. **Define Violation Zones**: Set up virtual zones around the intersection where vehicles shouldn't pass during red lights
4. **Real-time Violation Detection**: Alert when vehicles cross into violation zones while the traffic light is red

## How It Works

### Traffic Light State Detection

The system uses computer vision techniques to analyze the traffic light's color:

```python
# Extract traffic light region from the detected bounding box
tl_region = frame[y1:y2, x1:x2]

# Convert to HSV color space for better color detection
hsv = cv2.cvtColor(tl_region, cv2.COLOR_BGR2HSV)

# Define color ranges for red, yellow, green
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
# ... similar for yellow and green

# Determine dominant color based on pixel counts
```

### Vehicle Tracking

The system tracks individual vehicles using a simple proximity-based tracking algorithm:

- Associates new detections with existing tracks based on distance
- Maintains position history for each vehicle
- Removes old tracks automatically

### Violation Detection Logic

```python
def check_violations(self, traffic_light_states):
    # 1. Check if any traffic light is red
    any_red_light = any(state == TrafficLightState.RED 
                       for state in traffic_light_states.values())
    
    # 2. For each vehicle track
    for track in self.vehicle_tracks:
        # 3. Check if vehicle entered violation zone during red light
        if self._entered_zone_during_red(track, zone, traffic_light_states):
            # 4. Record violation
            violation = {
                'track_id': track_id,
                'timestamp': time.time(),
                'position': current_pos,
                'vehicle_class': track.class_name,
                'traffic_light_state': 'RED'
            }
```

## Key Features

### 1. Real-time Traffic Light State Detection
- **Color Analysis**: Uses HSV color space for robust color detection
- **State Classification**: Identifies Red, Yellow, Green, or Unknown states
- **Visual Feedback**: Draws colored borders around traffic lights based on their state

### 2. Violation Zone Management
- **Automatic Setup**: Creates default violation zones based on typical intersection layout
- **Customizable Zones**: Can be configured for specific camera angles and intersections
- **Visual Indicators**: Shows violation zones as semi-transparent overlays

### 3. Vehicle Tracking
- **Multi-Vehicle Support**: Tracks multiple vehicles simultaneously
- **Class Recognition**: Distinguishes between cars, trucks, motorcycles, buses, bicycles
- **Trajectory Analysis**: Monitors vehicle movement patterns

### 4. Violation Alerts
- **Real-time Detection**: Immediate notification when violations occur
- **Visual Markers**: Red circles and text overlays on violating vehicles
- **Logging**: Detailed violation logs with timestamps and vehicle information

## Running the System

### Basic Usage

```bash
# Run the traffic violation detection system
python main_traffic_violation.py
```

### Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit the application |
| `s` | Save screenshots of current frames |
| `r` | Reset performance statistics |
| `z` | Reset violation zones and vehicle tracks |

### Display Information

The system displays real-time information on each camera feed:

- **FPS**: Current frames per second
- **Objects**: Number of detected objects
- **Detection Time**: Processing time in milliseconds
- **Violations**: Total violation count
- **Light Status**: Current traffic light states

## Configuration

### Violation Zone Setup

The system automatically creates violation zones based on typical intersection layouts:

```python
# Main intersection crossing zone
intersection_zone = ViolationZone(
    name="main_intersection",
    points=[
        (int(width * 0.2), int(height * 0.4)),   # Top-left
        (int(width * 0.8), int(height * 0.4)),   # Top-right
        (int(width * 0.8), int(height * 0.8)),   # Bottom-right
        (int(width * 0.2), int(height * 0.8))    # Bottom-left
    ]
)
```

### Traffic Light Detection Parameters

Adjust color detection sensitivity in `traffic_light_detector.py`:

```python
# Red color ranges (HSV)
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])

# Yellow color range
yellow_lower = np.array([15, 100, 100])
yellow_upper = np.array([35, 255, 255])

# Green color range
green_lower = np.array([40, 100, 100])
green_upper = np.array([80, 255, 255])
```

### Vehicle Classes

The system monitors these vehicle types for violations:

```python
vehicle_classes = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle'
}
```

## System Architecture

```
main_traffic_violation.py
├── TrafficViolationSystem (Main coordinator)
├── ObjectDetector (YOLO-based detection)
├── TrafficLightViolationDetector (Violation logic)
│   ├── Traffic light state detection
│   ├── Vehicle tracking
│   ├── Violation zone management
│   └── Violation checking
└── MultiCameraManager (Camera handling)
```

## Performance Considerations

### Optimization Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster inference
2. **Model Selection**: Use appropriate YOLO model size (nano/small/medium) based on accuracy vs speed requirements
3. **Resolution**: Adjust camera resolution based on detection needs
4. **Tracking**: Simple proximity-based tracking is fast but may lose tracks in crowded scenes

### Accuracy Improvements

1. **Camera Positioning**: Position cameras to have clear view of traffic lights
2. **Lighting Conditions**: System works best in good lighting conditions
3. **Calibration**: Fine-tune color ranges for specific traffic light types
4. **Zone Configuration**: Adjust violation zones based on actual intersection layout

## Troubleshooting

### Common Issues

1. **Traffic Light Not Detected**
   - Ensure YOLO model includes 'traffic light' class
   - Check if traffic light is clearly visible in frame
   - Verify lighting conditions

2. **Incorrect State Detection**
   - Adjust HSV color ranges in `detect_traffic_light_state()`
   - Check for reflections or shadows on traffic light
   - Ensure sufficient resolution for color analysis

3. **False Violations**
   - Verify violation zone placement
   - Check tracking accuracy
   - Adjust timing thresholds

4. **Performance Issues**
   - Reduce frame resolution
   - Use smaller YOLO model
   - Optimize tracking parameters

## Example Output

When running the system, you'll see:

```
🚦 Traffic Light Violation Detection System v1.0
===============================================
Features:
- Real-time traffic light state detection
- Vehicle tracking and violation monitoring
- Multi-camera support (RTSP + local cameras)
- Real-time YOLOv11 object detection
- GPU acceleration (if available)

Detection Capabilities:
- Traffic light color detection (Red/Yellow/Green)
- Vehicle movement tracking
- Red light violation detection
- Violation zone monitoring

Controls:
- Press 'q' or ESC to quit
- Press 's' to save screenshots
- Press 'r' to reset statistics
- Press 'z' to reset violation zones
```

## Future Enhancements

1. **Advanced Tracking**: Implement more sophisticated tracking algorithms (DeepSORT, etc.)
2. **Multiple Zones**: Support for multiple intersection zones with different traffic lights
3. **Time-based Rules**: Consider yellow light timing and grace periods
4. **Database Integration**: Store violations in database for analysis
5. **Alert System**: Send notifications or alerts for violations
6. **Speed Detection**: Combine with speed estimation for additional traffic monitoring

## Integration with Existing System

The traffic violation detection system builds upon your existing object detection infrastructure:

- Uses the same `Config` class for settings
- Leverages existing `ObjectDetector` and `MultiCameraManager`
- Maintains the same camera stream format and window management
- Adds violation-specific functionality as an overlay

This ensures compatibility with your current setup while adding powerful traffic monitoring capabilities.
