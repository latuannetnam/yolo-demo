#  Reference: https://github.com/Shrimpstanot/traffic_analysis/tree/main
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from typing import Dict, List
import time
import signal
import sys
from config import Config
from camera_stream import MultiCameraManager
from loguru import logger
from supervision.annotators.utils import ColorLookup

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])


class VideoProcessor:
    """
    Processes a video using a YOLO model and tracks objects across frames.

    Attributes:
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): IOU threshold for detections.
        model (YOLO): YOLO model for detections.
        tracker (sv.ByteTrack): Tracker for object tracking.
        zones_in (List[sv.PolygonZone]): Polygon zones for entering objects.
        zone_annotators (List[sv.PolygonZoneAnnotator]): Annotators for zones.
        box_annotator (sv.BoxAnnotator): Box annotator for drawing bounding boxes.
        label_annotator (sv.LabelAnnotator): Label annotator for drawing labels.
        trace_annotator (sv.TraceAnnotator): Trace annotator for drawing object traces.
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

        SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
        self.CLASS_NAMES_DICT = self.model.names
        self.SELECTED_CLASS_IDS = [
            {value: key for key, value in self.CLASS_NAMES_DICT.items()}[class_name]
            for class_name in SELECTED_CLASS_NAMES
        ]

        logger.info(f"Selected class IDs: {self.SELECTED_CLASS_IDS}")
        for class_id in self.SELECTED_CLASS_IDS:
            logger.info(f"Class ID {class_id}: {self.CLASS_NAMES_DICT[class_id]}")

        self.camera_manager = MultiCameraManager()
        self.running = False

        self.frame_processing_times = {}
        self.avg_processing_time = 0.0
        self.frame_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._initialize_zones()

        self.box_annotator = sv.EllipseAnnotator(color_lookup=ColorLookup.INDEX)
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK, color_lookup=ColorLookup.INDEX
        )
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.CENTER,
            trace_length=100,
            thickness=1,
            color_lookup=ColorLookup.INDEX,
        )

    def _initialize_zones(self):
        """Initializes polygon zones and annotators."""
        ZONE_IN_POLYGONS = Config.get_zone_in_polygons()
        self.zones_in = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.CENTER,),
            )
            for polygon in ZONE_IN_POLYGONS
        ]
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=COLORS.colors[i % len(COLORS.colors)],
                text_color=sv.Color.WHITE,
                text_scale=0.5,
                text_thickness=1,
                text_padding=5,
            )
            for i, zone in enumerate(self.zones_in)
        ]

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()

    def initialize_cameras(self) -> bool:
        """Initialize camera streams."""
        try:
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
            processed_frame, _ = self.process_frame(frame)
            filename = f"traffic_analysis_camera_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            logger.info(f"Screenshot saved: {filename}")

    def _reset_counts(self) -> None:
        """Reset zone counts and timing statistics."""
        self._initialize_zones()
        self.frame_processing_times = {}
        self.avg_processing_time = 0.0
        self.frame_count = 0
        logger.info("Zone counts and timing statistics reset.")

    def stop(self) -> None:
        """Stop the traffic analysis system."""
        self.running = False
        if hasattr(self, "camera_manager"):
            self.camera_manager.stop_all()
        cv2.destroyAllWindows()
        logger.info("Traffic analysis system stopped")

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Annotates a frame with detections and entry zones."""
        annotated_frame = frame.copy()

        for i, zone_annotator in enumerate(self.zone_annotators):
            annotated_frame = zone_annotator.annotate(
                scene=annotated_frame, label=f"Zone {i}"
            )

        labels = []
        if len(detections) > 0 and detections.confidence is not None:
            # Create labels with confidence
            labels = [f"{confidence:0.2f}" for confidence in detections.confidence]

            # Add class names if available
            if detections.class_id is not None:
                labels = [
                    f"{self.CLASS_NAMES_DICT.get(class_id, 'Unknown')} {label}"
                    for class_id, label in zip(detections.class_id, labels)
                ]

            # Add tracker IDs if available
            if detections.tracker_id is not None:
                labels = [
                    f"#{tracker_id} {label}"
                    for tracker_id, label in zip(detections.tracker_id, labels)
                ]

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        if labels:
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels
            )
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        """Processes a frame using the YOLO model and tracker for entry detection only."""
        try:
            results = self.model(
                frame,
                # imgsz=640,
                half=True,
                agnostic_nms=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.SELECTED_CLASS_IDS,
                verbose=False,
            )
            detections: sv.Detections = sv.Detections.from_ultralytics(results[0])
            detections = self.tracker.update_with_detections(detections)

            for zone in self.zones_in:
                zone.trigger(detections=detections)

            annotated_frame = self.annotate_frame(frame, detections)
            return annotated_frame, {}

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, {}

    def process_video(self):
        """Processes frames from camera stream using the YOLO model and tracker."""
        if not self.initialize_cameras():
            return

        self.running = True
        logger.info("Starting traffic analysis system...")

        active_cameras = self.camera_manager.get_active_cameras()
        for camera_id in active_cameras:
            window_name = f"Camera {camera_id} - Traffic Analysis"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
            x, y = self.camera_manager.get_window_position(camera_id)
            cv2.moveWindow(window_name, x, y)

        try:
            while self.running:
                frames = self.camera_manager.get_all_frames()
                if not frames:
                    time.sleep(0.01)
                    continue

                for camera_id, frame in frames.items():
                    frame_start_time = time.time()

                    if (
                        frame.shape[1] != Config.WINDOW_WIDTH
                        or frame.shape[0] != Config.WINDOW_HEIGHT
                    ):
                        frame = cv2.resize(
                            frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
                        )

                    annotated_frame, detailed_timing = self.process_frame(frame)

                    frame_processing_time = (time.time() - frame_start_time) * 1000

                    self.frame_processing_times[camera_id] = frame_processing_time
                    self.frame_count += 1
                    self.avg_processing_time = (
                        (self.avg_processing_time * (self.frame_count - 1))
                        + frame_processing_time
                    ) / self.frame_count

                    if Config.DISPLAY_FPS:
                        fps = self.camera_manager.get_camera_fps(camera_id)
                        y_offset = 30
                        line_height = 25

                        info_text = f"Camera {camera_id} | FPS: {fps:.1f}"
                        cv2.putText(
                            annotated_frame,
                            info_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        y_offset += line_height

                        timing_text = f"Process Time: {frame_processing_time:.1f}ms"
                        cv2.putText(
                            annotated_frame,
                            timing_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )
                        y_offset += line_height

                        avg_timing_text = f"Avg Time: {self.avg_processing_time:.1f}ms"
                        cv2.putText(
                            annotated_frame,
                            avg_timing_text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )

                        if detailed_timing:
                            timing_details = [
                                f"Inference: {detailed_timing.get('inference', 0):.1f}ms",
                                f"Tracking: {detailed_timing.get('tracking', 0):.1f}ms",
                                f"Zones: {detailed_timing.get('zones', 0):.1f}ms",
                                f"Draw: {detailed_timing.get('annotation', 0):.1f}ms",
                                f"Total: {detailed_timing.get('total', 0):.1f}ms",
                            ]
                            right_x = Config.WINDOW_WIDTH - 200
                            detail_y_offset = 80
                            detail_line_height = 20
                            for timing_text in timing_details:
                                cv2.putText(
                                    annotated_frame,
                                    timing_text,
                                    (right_x, detail_y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (0, 255, 255),
                                    1,
                                )
                                detail_y_offset += detail_line_height

                    window_name = f"Camera {camera_id} - Traffic Analysis"
                    cv2.imshow(window_name, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("s"):
                    self._save_screenshots(frames)
                elif key == ord("r"):
                    self._reset_counts()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()


if __name__ == "__main__":
    print("🚦 Real-time Entry Detection System v2.0")
    print("=========================================")
    print("Features:")
    print("- Multi-camera support (RTSP + local cameras)")
    print("- Real-time entry zone analysis with supervision")
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

