#  Reference: https://github.com/Shrimpstanot/traffic_analysis/tree/main
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from typing import Dict, Iterable, List, Set
import numpy as np
import time
import signal
import sys
from config import Config
from camera_stream import MultiCameraManager
from loguru import logger
from  supervision.annotators.utils import ColorLookup
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

# Load zone configurations from environment variables via Config

# ZONE_OUT_POLYGONS = Config.get_zone_out_polygons()


class EntryDetectionManager:
    """
    Manages entry detections and counts objects entering zones.

    Attributes:
        tracked_objects (Set[int]): Set of tracker IDs that have been counted.
        entry_counts (Dict[int, int]): Count of objects entered per zone.
        zone_object_counts (Dict[int, Set[int]]): Objects currently in each zone.
    """

    def __init__(self) -> None:
        self.tracked_objects: Set[int] = set()
        self.entry_counts: Dict[int, int] = {}
        self.zone_object_counts: Dict[int, Set[int]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
    ) -> sv.Detections:
        """
        Updates the entry detection manager with new detections.

        Args:
            detections_all (sv.Detections): All detections in the frame.
            detections_in_zones (List[sv.Detections]): Detections in each entry zone.

        Returns:
            sv.Detections: All detections for display.
        """
        # Process each entry zone
        for zone_id, detections_in_zone in enumerate(detections_in_zones):
            try:
                # Initialize zone tracking if not exists
                if zone_id not in self.entry_counts:
                    self.entry_counts[zone_id] = 0
                    self.zone_object_counts[zone_id] = set()
                
                # Get current objects in this zone
                current_objects = set()
                if detections_in_zone.tracker_id is not None:
                    try:
                        # Ensure we have a valid array-like object
                        tracker_ids = np.asarray(detections_in_zone.tracker_id)
                        if tracker_ids.size > 0:
                            # Convert to integers, filtering out invalid values
                            valid_ids = []
                            for tid in tracker_ids.flatten():
                                try:
                                    if not np.isnan(tid) and not np.isinf(tid):
                                        valid_ids.append(int(tid))
                                except (ValueError, TypeError, OverflowError):
                                    continue
                            current_objects = set(valid_ids)
                    except Exception as convert_error:
                        logger.error(f"Error converting tracker_id in zone {zone_id}: {convert_error}")
                        logger.debug(f"tracker_id type: {type(detections_in_zone.tracker_id)}")
                        current_objects = set()
                
                # Find new objects that entered this zone
                previous_objects = self.zone_object_counts[zone_id]
                new_entries = current_objects - previous_objects
                
                # Count new entries (objects that weren't previously tracked globally)
                for tracker_id in new_entries:
                    if tracker_id not in self.tracked_objects:
                        self.entry_counts[zone_id] += 1
                        self.tracked_objects.add(tracker_id)
                        logger.info(f"New entry in Zone {zone_id}: Object #{tracker_id} - Total: {self.entry_counts[zone_id]}")
                
                # Update zone object tracking
                self.zone_object_counts[zone_id] = current_objects
                
            except Exception as e:
                logger.error(f"Error processing zone {zone_id}: {e}")
                logger.debug(f"detections_in_zone.tracker_id type: {type(detections_in_zone.tracker_id)}")
                if detections_in_zone.tracker_id is not None:
                    logger.debug(f"tracker_id values: {detections_in_zone.tracker_id}")
                continue

        return detections_all

    def get_total_entries(self) -> int:
        """Get total entries across all zones."""
        return sum(self.entry_counts.values())
    
    def get_zone_entries(self, zone_id: int) -> int:
        """Get entries for a specific zone."""
        return self.entry_counts.get(zone_id, 0)
    
    def reset(self) -> None:
        """Reset all counts and tracking."""
        self.tracked_objects.clear()
        self.entry_counts.clear()
        self.zone_object_counts.clear()


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    """Creates polygon zones from polygons and triggering anchors."""
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    """
    Processes a video using a YOLO model and tracks objects across frames.

    Attributes:
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IOU threshold for detections.
        model (YOLO): YOLO model for detections.
        tracker (sv.ByteTrack): Tracker for object tracking.
        video_info (sv.VideoInfo): Video information.
        zones_in (List[sv.PolygonZone]): Polygon zones for entering objects.
        box_annotator (sv.BoxAnnotator): Box annotator for drawing bounding boxes.
        label_annotator (sv.LabelAnnotator): Label annotator for drawing labels.
        trace_annotator (sv.TraceAnnotator): Trace annotator for drawing object traces.
        detections_manager (DetectionsManager): Detections manager for tracking objects.
    """

    def __init__(
        self,
        source_weights_path: str,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        
        # the class names we have chosen
        SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']

        # dict maping class_id to class_name
        self.CLASS_NAMES_DICT = self.model.names
        

        # class ids matching the class names we have chosen
        self.SELECTED_CLASS_IDS = [
            {value: key for key, value in self.CLASS_NAMES_DICT.items()}[class_name]
            for class_name
            in SELECTED_CLASS_NAMES
        ]

        logger.info(f"Selected class IDs: {self.SELECTED_CLASS_IDS}")
        for class_id in self.SELECTED_CLASS_IDS:
            logger.info(f"Class ID {class_id}: {self.CLASS_NAMES_DICT[class_id]}")

        # Initialize camera manager
        self.camera_manager = MultiCameraManager()
        self.running = False
        
        # Timing statistics
        self.frame_processing_times = {}  # Store processing times per camera
        self.avg_processing_time = 0.0
        self.frame_count = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Only create entry zones (no exit zones needed)
        ZONE_IN_POLYGONS = Config.get_zone_in_polygons()
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])

        # self.box_annotator = sv.BoxAnnotator(color=COLORS)
        # self.label_annotator = sv.LabelAnnotator(
        #     color=COLORS, text_color=sv.Color.BLACK
        # )

        # self.box_annotator = sv.BoxAnnotator(color_lookup=ColorLookup.INDEX)
        self.box_annotator = sv.EllipseAnnotator(color_lookup=ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            color_lookup=ColorLookup.INDEX
        )
        
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.CENTER, trace_length=100, thickness=1, color_lookup=ColorLookup.INDEX
        )
        self.entry_manager = EntryDetectionManager()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()

    def initialize_cameras(self) -> bool:
        """Initialize camera streams."""
        try:
            # Setup cameras
            logger.info("Setting up cameras...")
            streams = Config.get_stream_config()
            
            for i, stream in enumerate(streams):
                if self.camera_manager.add_camera(stream, i):
                    logger.info(f"Camera {i} added successfully")
                else:
                    logger.error(f"Failed to add camera {i}: {stream}")
            
            active_cameras = self.camera_manager.get_active_cameras()
            if not active_cameras:
                logger.error("No cameras available. Exiting.")
                return False
            
            logger.info(f"System initialized with {len(active_cameras)} cameras")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cameras: {e}")
            return False

   
    def _save_screenshots(self, frames: Dict[int, np.ndarray]) -> None:
        """Save screenshots of current frames."""
        timestamp = int(time.time())
        for camera_id, frame in frames.items():
            processed_frame, _ = self.process_frame(frame)  # Unpack tuple, ignore timing
            filename = f"traffic_analysis_camera_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Screenshot saved: {filename}")
    
    def _reset_detection_manager(self) -> None:
        """Reset entry detection manager and timing statistics."""
        self.entry_manager.reset()
        # Reset timing statistics
        self.frame_processing_times = {}
        self.avg_processing_time = 0.0
        self.frame_count = 0
        logger.info("Entry detection manager and timing statistics reset - all counts cleared")

    def stop(self) -> None:
        """Stop the traffic analysis system."""
        self.running = False
        
        # Stop all cameras
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop_all()
        
        # Close all windows
        cv2.destroyAllWindows()
        
        logger.info("Traffic analysis system stopped")

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """
        Annotates a frame with detections and entry zones.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections to annotate.

        Returns:
            np.ndarray: The annotated frame.
        """
        annotated_frame = frame.copy()
        
        # Draw entry zones only
        for i, zone_in in enumerate(self.zones_in):
            try:
                color_index = i % len(COLORS.colors)
                annotated_frame = sv.draw_polygon(
                    annotated_frame, zone_in.polygon, COLORS.colors[color_index]
                )
            except Exception as e:
                logger.error(f"Error drawing polygon for zone {i}: {e}")
                continue

        # Draw detection boxes and labels
        labels = []
        if len(detections) > 0 and detections.confidence is not None and detections.class_id is not None:
            try:
                # Create labels with class name and confidence
                # labels = [
                #     f"{class_id}:{self.CLASS_NAMES_DICT[class_id]}:{confidence:0.2f}"
                #     for confidence, class_id in zip(detections.confidence, detections.class_id)
                # ]

                labels = [
                    f"{confidence:0.2f}"
                    for confidence in detections.confidence
                ]
                
                # Add tracker IDs if available
                if detections.tracker_id is not None:
                    try:
                        tracker_ids = np.asarray(detections.tracker_id)
                        if tracker_ids.size > 0 and len(tracker_ids) == len(labels):
                            labels = [
                                f"#{int(tid)} {label}" if not np.isnan(tid) and not np.isinf(tid) else f"#? {label}"
                                for label, tid in zip(labels, tracker_ids.flatten())
                            ]
                    except Exception as tracker_error:
                        logger.error(f"Error adding tracker IDs to labels: {tracker_error}")
                        
            except Exception as e:
                logger.error(f"Error creating labels: {e}")
                labels = []
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        if labels:
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels
            )

        # Display entry counts for each zone
        for zone_id, zone_in in enumerate(self.zones_in):
            try:
                zone_center = sv.get_polygon_center(polygon=zone_in.polygon)
                entry_count = self.entry_manager.get_zone_entries(zone_id)
                
                # Ensure coordinates are integers
                center_x = int(zone_center.x)
                center_y = int(zone_center.y)
                
                # Display zone entry count
                text_anchor = sv.Point(x=center_x, y=center_y)
                color_index = zone_id % len(COLORS.colors)
                annotated_frame = sv.draw_text(
                    scene=annotated_frame,
                    text=f"Zone {zone_id}: {entry_count}",
                    text_anchor=text_anchor,
                    background_color=COLORS.colors[color_index],
                    text_color=sv.Color.WHITE
                )
            except Exception as e:
                logger.error(f"Error drawing zone {zone_id} text: {e}")
                continue

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Processes a frame using the YOLO model and tracker for entry detection only."""
        try:
            # Time YOLO inference
            inference_start = time.time()
            results = self.model(
                frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            inference_time = (time.time() - inference_start) * 1000
            # only consider class id from selected_classes define above
            detections = detections[np.isin(detections.class_id, self.SELECTED_CLASS_IDS)]
            
            # Time tracking
            tracking_start = time.time()
            # Ensure class_id is properly initialized
            if len(detections) > 0:
                detections.class_id = np.zeros(len(detections), dtype=np.int32)
            else:
                detections.class_id = np.array([], dtype=np.int32)
                
            detections = self.tracker.update_with_detections(detections)
            tracking_time = (time.time() - tracking_start) * 1000

            # Time zone processing
            zone_start = time.time()
            # Only process entry zones
            detections_in_zones = []
            for zone_in in self.zones_in:
                try:
                    # Get boolean mask for detections in zone
                    zone_mask = zone_in.trigger(detections=detections)
                    # Apply mask to get detections in this zone
                    detections_in_zone = detections[zone_mask]
                    detections_in_zones.append(detections_in_zone)
                except Exception as zone_error:
                    logger.error(f"Error processing zone trigger: {zone_error}")
                    # Add empty detection for this zone
                    empty_detections = sv.Detections.empty()
                    detections_in_zones.append(empty_detections)

            # Update entry manager with detections
            detections = self.entry_manager.update(detections, detections_in_zones)
            zone_time = (time.time() - zone_start) * 1000
            
            # Time annotation
            annotation_start = time.time()
            result_frame = self.annotate_frame(frame, detections)
            annotation_time = (time.time() - annotation_start) * 1000
            
            # Prepare timing information for display in process_video
            total_time = inference_time + tracking_time + zone_time + annotation_time
            timing_info = {
                'inference': inference_time,
                'tracking': tracking_time,
                'zones': zone_time,
                'annotation': annotation_time,
                'total': total_time
            }
            
            return result_frame, timing_info
            
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
            # Return the original frame and empty timing if processing fails
            return frame, {}
    
    def process_video(self):
        """
        Processes frames from camera stream using the YOLO model and tracker.
        """
        if not self.initialize_cameras():
            return
        
        self.running = True
        logger.info("Starting traffic analysis system...")
        
        # Create windows for each camera
        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Traffic Analysis"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
            
            # Position windows
            x, y = self.camera_manager.get_window_position(camera_id)
            cv2.moveWindow(window_name, x, y)
        
        try:
            while self.running:
                # Get frames from all cameras
                frames = self.camera_manager.get_all_frames()
                
                if not frames:
                    time.sleep(0.01)
                    continue
                
                # Process each frame
                for camera_id, frame in frames.items():
                    # Start timing
                    frame_start_time = time.time()
                    
                    # Resize frame if necessary
                    if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                        frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                    
                    # Process frame with traffic analysis
                    annotated_frame, detailed_timing = self.process_frame(frame)
                    
                    # Calculate total processing time
                    frame_processing_time = (time.time() - frame_start_time) * 1000  # Convert to milliseconds
                    
                    # Update timing statistics
                    self.frame_processing_times[camera_id] = frame_processing_time
                    self.frame_count += 1
                    self.avg_processing_time = ((self.avg_processing_time * (self.frame_count - 1)) + frame_processing_time) / self.frame_count
                    
                    # Get total entries for display
                    total_entries = self.entry_manager.get_total_entries()
                    
                    # Add all text overlays with proper spacing
                    if Config.DISPLAY_FPS:
                        fps = self.camera_manager.get_camera_fps(camera_id)
                        
                        # Left side text (top to bottom with proper spacing)
                        y_offset = 30
                        line_height = 25
                        
                        # Camera info
                        info_text = f"Camera {camera_id} | FPS: {fps:.1f}"
                        cv2.putText(annotated_frame, info_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += line_height
                        
                        # Current frame processing time
                        timing_text = f"Process Time: {frame_processing_time:.1f}ms"
                        cv2.putText(annotated_frame, timing_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_offset += line_height
                        
                        # Average processing time
                        avg_timing_text = f"Avg Time: {self.avg_processing_time:.1f}ms"
                        cv2.putText(annotated_frame, avg_timing_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += line_height
                        
                        # # Total entries
                        # entries_text = f"Total Entries: {total_entries}"
                        # cv2.putText(annotated_frame, entries_text, (10, y_offset), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Right side detailed timing breakdown
                        if detailed_timing:
                            timing_details = [
                                f"Inference: {detailed_timing.get('inference', 0):.1f}ms",
                                f"Tracking: {detailed_timing.get('tracking', 0):.1f}ms", 
                                f"Zones: {detailed_timing.get('zones', 0):.1f}ms",
                                f"Draw: {detailed_timing.get('annotation', 0):.1f}ms",
                                f"Total: {detailed_timing.get('total', 0):.1f}ms"
                            ]
                            
                            # Draw detailed timing on right side
                            right_x = Config.WINDOW_WIDTH - 200
                            detail_y_offset = 80
                            detail_line_height = 20
                            
                            for timing_text in timing_details:
                                cv2.putText(annotated_frame, timing_text, (right_x, detail_y_offset), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                                detail_y_offset += detail_line_height
                    
                    # Display frame
                    window_name = f"Camera {camera_id} - Traffic Analysis"
                    cv2.imshow(window_name, annotated_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s'):  # 's' key to save screenshot
                    self._save_screenshots(frames)
                elif key == ord('r'):  # 'r' key to reset detection manager
                    self._reset_detection_manager()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()




if __name__ == "__main__":
    print("🚦 Real-time Entry Detection System v1.0")
    print("=========================================")
    print("Features:")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time entry zone analysis")
    print("- Object counting (entry-only mode)")
    print("- GPU acceleration (if available)")
    print()
    print("Controls:")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save screenshots")
    print("- Press 'r' to reset entry counts")
    print()
    
    processor = VideoProcessor(
        source_weights_path=Config.get_full_model_name(),
        confidence_threshold=Config.MODEL_CONFIDENCE,
        iou_threshold=Config.MODEL_IOU_THRESHOLD,
    )
    
    try:
        processor.process_video()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

# Instructions for entry detection:
# 1. Run the application.
# 2. The system will automatically detect and connect to available cameras.
# 3. Entry zones are predefined for object counting (vehicles, people, etc.).
# 4. Press 'q' or ESC to quit the application.
# 5. Press 's' to save screenshots of all camera feeds.
# 6. Press 'r' to reset entry counts and start fresh counting.
# 7. The system displays real-time entry counts per zone and total entries.

