"""
Test script for traffic light violation detection system.
"""
import cv2
import numpy as np
import time
from traffic_light_detector import TrafficLightViolationDetector, TrafficLightState

def create_test_frame_with_traffic_light():
    """Create a test frame with a simulated traffic light."""
    # Create a test frame (simulating street intersection)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road
    cv2.rectangle(frame, (0, 200), (640, 280), (64, 64, 64), -1)  # Horizontal road
    cv2.rectangle(frame, (280, 0), (360, 480), (64, 64, 64), -1)  # Vertical road
    
    # Draw crosswalk lines
    for i in range(0, 640, 20):
        cv2.rectangle(frame, (i, 200), (i+10, 210), (255, 255, 255), -1)
        cv2.rectangle(frame, (i, 270), (i+10, 280), (255, 255, 255), -1)
    
    for i in range(0, 480, 20):
        cv2.rectangle(frame, (280, i), (290, i+10), (255, 255, 255), -1)
        cv2.rectangle(frame, (350, i), (360, i+10), (255, 255, 255), -1)
    
    # Draw traffic light pole
    cv2.rectangle(frame, (100, 50), (110, 200), (128, 128, 128), -1)
    
    # Draw traffic light box
    cv2.rectangle(frame, (80, 50), (130, 120), (64, 64, 64), -1)
    
    return frame

def simulate_red_traffic_light(frame):
    """Add a red traffic light to the frame."""
    # Red light (top circle)
    cv2.circle(frame, (105, 65), 12, (0, 0, 255), -1)
    # Dark yellow light
    cv2.circle(frame, (105, 85), 12, (0, 64, 64), -1)
    # Dark green light
    cv2.circle(frame, (105, 105), 12, (0, 64, 0), -1)
    return frame

def simulate_green_traffic_light(frame):
    """Add a green traffic light to the frame."""
    # Dark red light
    cv2.circle(frame, (105, 65), 12, (0, 0, 64), -1)
    # Dark yellow light
    cv2.circle(frame, (105, 85), 12, (0, 64, 64), -1)
    # Green light (bottom circle)
    cv2.circle(frame, (105, 105), 12, (0, 255, 0), -1)
    return frame

def simulate_vehicle_detections(is_violating=False):
    """Create simulated vehicle detections."""
    detections = []
    
    if is_violating:
        # Vehicle in violation zone during red light
        detections.append({
            'bbox': (300, 220, 380, 260),
            'confidence': 0.85,
            'class_id': 2,  # Car class
            'class_name': 'car'
        })
    
    # Other vehicles not violating
    detections.append({
        'bbox': (150, 220, 220, 260),
        'confidence': 0.92,
        'class_id': 2,
        'class_name': 'car'
    })
    
    detections.append({
        'bbox': (450, 220, 520, 260),
        'confidence': 0.78,
        'class_id': 7,  # Truck class
        'class_name': 'truck'
    })
    
    # Add traffic light detection
    detections.append({
        'bbox': (80, 50, 130, 120),
        'confidence': 0.95,
        'class_id': 9,  # Traffic light class
        'class_name': 'traffic light'
    })
    
    return detections

def test_traffic_light_violation_detector():
    """Test the traffic light violation detection system."""
    print("Testing Traffic Light Violation Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = TrafficLightViolationDetector()
    
    # Create test frame
    base_frame = create_test_frame_with_traffic_light()
    
    # Setup violation zones
    detector.setup_violation_zones_from_image(base_frame.shape[:2])
    
    print(f"Setup {len(detector.violation_zones)} violation zones")
    
    # Test scenario 1: Green light - no violations
    print("\nScenario 1: Green light with vehicles")
    frame1 = base_frame.copy()
    frame1 = simulate_green_traffic_light(frame1)
    detections1 = simulate_vehicle_detections(is_violating=False)
    
    # Update tracking
    detector.update_vehicle_tracks(detections1)
    
    # Draw zones and detections
    frame1 = detector.draw_violation_zones(frame1)
    frame1, traffic_states1 = detector.draw_traffic_light_states(frame1, detections1)
    
    # Check violations
    violations1 = detector.check_violations(traffic_states1)
    print(f"Traffic light states: {[state.value for state in traffic_states1.values()]}")
    print(f"Violations detected: {len(violations1)}")
    
    # Test scenario 2: Red light with violation
    print("\nScenario 2: Red light with vehicle in intersection")
    frame2 = base_frame.copy()
    frame2 = simulate_red_traffic_light(frame2)
    detections2 = simulate_vehicle_detections(is_violating=True)
    
    # Simulate time passage
    time.sleep(0.1)
    
    # Update tracking
    detector.update_vehicle_tracks(detections2)
    
    # Draw zones and detections
    frame2 = detector.draw_violation_zones(frame2)
    frame2, traffic_states2 = detector.draw_traffic_light_states(frame2, detections2)
    
    # Check violations
    violations2 = detector.check_violations(traffic_states2)
    frame2 = detector.draw_violations(frame2, violations2)
    
    print(f"Traffic light states: {[state.value for state in traffic_states2.values()]}")
    print(f"Violations detected: {len(violations2)}")
    
    if violations2:
        for violation in violations2:
            print(f"  - {violation['vehicle_class']} violated at position {violation['position']}")
    
    # Display results
    print("\nDisplaying test results...")
    print("Press any key to close windows")
    
    cv2.namedWindow("Green Light - No Violations", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red Light - With Violation", cv2.WINDOW_NORMAL)
    
    cv2.imshow("Green Light - No Violations", frame1)
    cv2.imshow("Red Light - With Violation", frame2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nTest completed!")
    print(f"Total vehicle tracks: {len(detector.vehicle_tracks)}")
    print(f"Total violations recorded: {len(detector.violations)}")

def test_traffic_light_state_detection():
    """Test traffic light state detection specifically."""
    print("\nTesting Traffic Light State Detection")
    print("-" * 40)
    
    detector = TrafficLightViolationDetector()
    
    # Test with different colored traffic lights
    test_cases = [
        ("Red Light", simulate_red_traffic_light),
        ("Green Light", simulate_green_traffic_light),
    ]
    
    for case_name, light_function in test_cases:
        frame = create_test_frame_with_traffic_light()
        frame = light_function(frame)
        
        # Traffic light bounding box
        traffic_light_bbox = (80, 50, 130, 120)
        
        # Detect state
        state = detector.detect_traffic_light_state(frame, traffic_light_bbox)
        
        print(f"{case_name}: Detected state = {state.value}")
        
        # Draw bounding box
        x1, y1, x2, y2 = traffic_light_bbox
        color = {
            TrafficLightState.RED: (0, 0, 255),
            TrafficLightState.GREEN: (0, 255, 0),
            TrafficLightState.YELLOW: (0, 255, 255),
            TrafficLightState.UNKNOWN: (128, 128, 128)
        }[state]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"State: {state.value}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow(f"Traffic Light State Test - {case_name}", frame)
        cv2.waitKey(2000)  # Show for 2 seconds
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚦 Traffic Light Violation Detection - Test Suite")
    print("=" * 55)
    
    try:
        # Test traffic light state detection
        test_traffic_light_state_detection()
        
        # Test full violation detection
        test_traffic_light_violation_detector()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
