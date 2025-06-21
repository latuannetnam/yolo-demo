"""
Traffic Light State Detection and Violation Monitoring System.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum

class TrafficLightState(Enum):
    """Traffic light state enumeration."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"

@dataclass
class ViolationZone:
    """Defines a violation zone for traffic light monitoring."""
    name: str
    points: List[Tuple[int, int]]  # Polygon points defining the zone
    traffic_light_bbox: Optional[Tuple[int, int, int, int]] = None  # Associated traffic light
    
@dataclass
class VehicleTrack:
    """Track individual vehicles for violation detection."""
    track_id: int
    positions: deque  # Recent positions
    class_name: str
    first_seen: float
    last_seen: float
    violated: bool = False

class TrafficLightViolationDetector:
    """Detects traffic light violations by monitoring vehicle movement."""
    
    def __init__(self):
        """Initialize the traffic light violation detector."""
        self.violation_zones: List[ViolationZone] = []
        self.vehicle_tracks: Dict[int, VehicleTrack] = {}
        self.next_track_id = 0
        self.violations: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_track_age = 5.0  # seconds
        self.max_track_distance = 100  # pixels
        self.position_history_length = 10
        
        # Traffic light state detection
        self.traffic_light_states: Dict[int, TrafficLightState] = {}
        
        # Vehicle classes that can violate traffic lights
        self.vehicle_classes = {
            'car', 'truck', 'bus', 'motorcycle', 'bicycle'
        }
        
    def detect_traffic_light_state(self, frame: np.ndarray, traffic_light_bbox: Tuple[int, int, int, int]) -> TrafficLightState:
        """
        Detect the state of a traffic light from its bounding box region.
        
        Args:
            frame: Input frame
            traffic_light_bbox: Bounding box of the traffic light (x1, y1, x2, y2)
            
        Returns:
            TrafficLightState enum value
        """
        try:
            x1, y1, x2, y2 = traffic_light_bbox
            
            # Extract traffic light region
            tl_region = frame[y1:y2, x1:x2]
            
            if tl_region.size == 0:
                return TrafficLightState.UNKNOWN
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(tl_region, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for red, yellow, green in HSV
            # Red ranges (handle wrap-around)
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            
            # Yellow range
            yellow_lower = np.array([15, 100, 100])
            yellow_upper = np.array([35, 255, 255])
            
            # Green range
            green_lower = np.array([40, 100, 100])
            green_upper = np.array([80, 255, 255])
            
            # Create masks
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            
            # Count pixels for each color
            red_pixels = cv2.countNonZero(red_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)
            green_pixels = cv2.countNonZero(green_mask)
            
            # Determine dominant color
            max_pixels = max(red_pixels, yellow_pixels, green_pixels)
            
            if max_pixels < 50:  # Minimum threshold
                return TrafficLightState.UNKNOWN
            
            if red_pixels == max_pixels:
                return TrafficLightState.RED
            elif yellow_pixels == max_pixels:
                return TrafficLightState.YELLOW
            elif green_pixels == max_pixels:
                return TrafficLightState.GREEN
            
            return TrafficLightState.UNKNOWN
            
        except Exception as e:
            logger.error(f"Error detecting traffic light state: {e}")
            return TrafficLightState.UNKNOWN
    
    def setup_violation_zones_from_image(self, frame_shape: Tuple[int, int]) -> None:
        """
        Setup violation zones based on typical traffic intersection layout.
        
        Args:
            frame_shape: (height, width) of the frame
        """
        height, width = frame_shape
        
        # Create default violation zones for a typical intersection
        # These would ideally be configured per camera/intersection
        
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
        
        self.violation_zones = [intersection_zone]
        logger.info(f"Setup {len(self.violation_zones)} violation zones")
    
    def is_point_in_zone(self, point: Tuple[int, int], zone: ViolationZone) -> bool:
        """
        Check if a point is inside a violation zone.
        
        Args:
            point: (x, y) coordinates
            zone: ViolationZone to check
            
        Returns:
            True if point is inside the zone
        """
        try:
            pts = np.array(zone.points, np.int32)
            return cv2.pointPolygonTest(pts, point, False) >= 0
        except Exception:
            return False
    
    def update_vehicle_tracks(self, detections: List[Dict[str, Any]]) -> None:
        """
        Update vehicle tracking for violation detection.
        
        Args:
            detections: List of detection dictionaries
        """
        current_time = time.time()
        
        # Extract vehicle detections
        vehicle_detections = [
            det for det in detections 
            if det['class_name'].lower() in self.vehicle_classes
        ]
        
        # Simple tracking by proximity (in production, use more sophisticated tracking)
        matched_tracks = set()
        
        for detection in vehicle_detections:
            x1, y1, x2, y2 = detection['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find closest existing track
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in self.vehicle_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                if track.positions:
                    last_pos = track.positions[-1]
                    distance = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                    
                    if distance < self.max_track_distance and distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.vehicle_tracks[best_track_id]
                track.positions.append(center)
                track.last_seen = current_time
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                new_track = VehicleTrack(
                    track_id=self.next_track_id,
                    positions=deque([center], maxlen=self.position_history_length),
                    class_name=detection['class_name'],
                    first_seen=current_time,
                    last_seen=current_time
                )
                self.vehicle_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.vehicle_tracks.items():
            if current_time - track.last_seen > self.max_track_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.vehicle_tracks[track_id]
    
    def check_violations(self, traffic_light_states: Dict[int, TrafficLightState]) -> List[Dict[str, Any]]:
        """
        Check for traffic light violations.
        
        Args:
            traffic_light_states: Dictionary mapping traffic light IDs to their states
            
        Returns:
            List of violation events
        """
        current_violations = []
        
        # Check if any traffic light is red
        any_red_light = any(state == TrafficLightState.RED for state in traffic_light_states.values())
        
        if not any_red_light:
            return current_violations
        
        # Check each vehicle track for violations
        for track_id, track in self.vehicle_tracks.items():
            if track.violated or len(track.positions) < 2:
                continue
            
            # Check if vehicle is in a violation zone
            current_pos = track.positions[-1]
            
            for zone in self.violation_zones:
                if self.is_point_in_zone(current_pos, zone):
                    # Check if vehicle entered the zone while light was red
                    if self._entered_zone_during_red(track, zone, traffic_light_states):
                        violation = {
                            'track_id': track_id,
                            'timestamp': time.time(),
                            'position': current_pos,
                            'zone_name': zone.name,
                            'vehicle_class': track.class_name,
                            'traffic_light_state': 'RED'
                        }
                        current_violations.append(violation)
                        track.violated = True
                        logger.warning(f"Traffic violation detected: {track.class_name} in {zone.name}")
        
        return current_violations
    
    def _entered_zone_during_red(self, track: VehicleTrack, zone: ViolationZone, 
                                traffic_light_states: Dict[int, TrafficLightState]) -> bool:
        """
        Check if vehicle entered the zone while the light was red.
        
        Args:
            track: Vehicle track
            zone: Violation zone
            traffic_light_states: Current traffic light states
            
        Returns:
            True if vehicle entered zone during red light
        """
        # Simple check: if we have at least 2 positions
        if len(track.positions) < 2:
            return False
        
        previous_pos = track.positions[-2]
        current_pos = track.positions[-1]
        
        # Check if vehicle was outside zone and is now inside
        was_outside = not self.is_point_in_zone(previous_pos, zone)
        is_inside = self.is_point_in_zone(current_pos, zone)
        
        # Vehicle entered the zone
        if was_outside and is_inside:
            # Check if any light is red (simplified - in reality, check specific light for this zone)
            return any(state == TrafficLightState.RED for state in traffic_light_states.values())
        
        return False
    
    def draw_violations(self, frame: np.ndarray, violations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw violation indicators on the frame.
        
        Args:
            frame: Input frame
            violations: List of violations to draw
            
        Returns:
            Frame with violation indicators
        """
        for violation in violations:
            x, y = violation['position']
            
            # Draw violation marker
            cv2.circle(frame, (x, y), 20, (0, 0, 255), 3)  # Red circle
            cv2.circle(frame, (x, y), 25, (0, 0, 255), 2)  # Outer red circle
            
            # Draw violation text
            text = f"VIOLATION: {violation['vehicle_class']}"
            cv2.putText(frame, text, (x - 50, y - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw timestamp
            time_text = f"Time: {violation['timestamp']:.1f}"
            cv2.putText(frame, time_text, (x - 50, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame
    
    def draw_violation_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw violation zones on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with violation zones drawn
        """
        for zone in self.violation_zones:
            pts = np.array(zone.points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw zone boundary
            cv2.polylines(frame, [pts], True, (255, 255, 0), 2)  # Yellow boundary
            
            # Fill zone with semi-transparent color
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (255, 255, 0))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            
            # Draw zone name
            if zone.points:
                center_x = int(np.mean([p[0] for p in zone.points]))
                center_y = int(np.mean([p[1] for p in zone.points]))
                cv2.putText(frame, f"Zone: {zone.name}", (center_x - 50, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def draw_traffic_light_states(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[int, TrafficLightState]]:
        """
        Draw traffic light states and return the detected states.
        
        Args:
            frame: Input frame
            detections: List of all detections
            
        Returns:
            Tuple of (frame with traffic light states, traffic light states dict)
        """
        traffic_light_states = {}
        
        # Find traffic light detections
        for i, detection in enumerate(detections):
            if detection['class_name'].lower() == 'traffic light':
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Detect traffic light state
                state = self.detect_traffic_light_state(frame, bbox)
                traffic_light_states[i] = state
                
                # Draw state indicator
                state_color = {
                    TrafficLightState.RED: (0, 0, 255),
                    TrafficLightState.YELLOW: (0, 255, 255),
                    TrafficLightState.GREEN: (0, 255, 0),
                    TrafficLightState.UNKNOWN: (128, 128, 128)
                }[state]
                
                # Draw thick border around traffic light based on state
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), state_color, 4)
                
                # Draw state text
                state_text = f"Light: {state.value.upper()}"
                cv2.putText(frame, state_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        return frame, traffic_light_states
