"""
Object detection module using YOLOv11.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
from typing import List, Tuple, Dict, Any
from loguru import logger
import supervision as sv
from supervision.annotators.utils import ColorLookup
import math

from config import Config

class ObjectDetector:
    """YOLOv11-based object detector that can handle both PyTorch and TensorRT models."""

    def __init__(self):
        """Initialize the object detector."""
        self.model = None
        self.device = self._setup_device()
        self.class_names: Dict[int, str] = {}
        self.model_name = ""
        self.model_type = ""
        
        self.slicer = None
        self.selected_class_names = Config.get_selected_class_names()
        self.selected_class_ids: List[int] = []        

        self.tracker = sv.ByteTrack()
        
        if Config.USE_SMOOTHER:
            self.smoother = sv.DetectionsSmoother()
        else:
            self.smoother = None

        self.box_annotator = sv.BoxAnnotator(color_lookup=ColorLookup.CLASS)
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK, color_lookup=ColorLookup.CLASS
        )
        if Config.USE_TRACE:
            self.trace_annotator = sv.TraceAnnotator(
                position=sv.Position.CENTER,
                trace_length=30,
                thickness=2,
                color_lookup=ColorLookup.CLASS,
            )
        else:
            self.trace_annotator = None

        if Config.USE_HEATMAP:
            self.heat_map_annotator = sv.HeatMapAnnotator()
        else:
            self.heat_map_annotator = None

        self.polygon_zones = []
        self.polygon_zone_annotators = []
        self.zone_tracker_ids = []
        if Config.USE_POLYGON_ZONE:
            polygons = Config.get_zone_in_polygons()
            # Initialize zones, trackers, and annotators in a single pass
            for polygon in polygons:
                zone = sv.PolygonZone(polygon=polygon)
                self.polygon_zones.append(zone)
                self.zone_tracker_ids.append(set())
                self.polygon_zone_annotators.append(
                    sv.PolygonZoneAnnotator(
                        zone=zone,
                        color=sv.Color.RED,
                        thickness=2,
                        text_thickness=1,
                        text_scale=0.5,
                    )
                )

        # Optimized initialization of line zones and annotators
        self.line_zones = []
        self.line_annotators = []

        if Config.USE_LINE_ZONE:
            for idx, (start_coords, end_coords) in enumerate(Config.LINE_ZONES):
                # Unpack points and create zones
                sp = sv.Point(*start_coords)
                ep = sv.Point(*end_coords)
                logger.info(f"Line zone {idx}: start=({sp.x}, {sp.y}), end=({ep.x}, {ep.y})")

                self.line_zones.append(sv.LineZone(start=sp, end=ep))
                self.line_annotators.append(sv.LineZoneAnnotator(color=sv.Color.RED,
                                                                 thickness=2,
                                                                 text_thickness=1,
                                                                 text_scale=0.5))

        self._load_model()

    def _calculate_slice_dimensions(self, frame_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate slice and overlap dimensions based on number of tiles in frame."""
        h, w = frame_shape
        num_tiles = Config.NUM_TILES

        if num_tiles <= 0:
            logger.error("Number of tiles must be a positive integer.")
            return (w, h), (0, 0)

        if num_tiles == 1:
            return (w, h), (0, 0)

        overlap_ratio_h = 0.2  # 20% vertical overlap

        # Formula to calculate slice height so that tiles cover the whole frame with overlap
        slice_h = h / (num_tiles - (num_tiles - 1) * overlap_ratio_h)
        overlap_h = slice_h * overlap_ratio_h

        slice_w = w
        overlap_w = 0

        logger.info(
            f"Calculating slice dimensions for {num_tiles} horizontal tiles with {overlap_ratio_h*100:.0f}% vertical overlap"
        )

        return (int(slice_w), int(math.ceil(slice_h))), (int(overlap_w), int(math.ceil(overlap_h)))

    def _init_slicer(self, frame_shape: Tuple[int, int]):
        """Initialize the inference slicer."""
        if not Config.USE_SLICER:
            self.slicer = None
            return
        slice_wh, overlap_wh = self._calculate_slice_dimensions(frame_shape)
        self.slicer = sv.InferenceSlicer(
            slice_wh=slice_wh,
            overlap_ratio_wh=None,
            overlap_wh=overlap_wh,
            callback=self._perform_inference,
            thread_workers=Config.SLICE_WORKERS,
        )
        logger.info(f"Initialized inference slicer with slice_wh={slice_wh}, overlap_wh={overlap_wh} and {Config.SLICE_WORKERS} workers")

    def _perform_inference(self, frame_slice: np.ndarray) -> sv.Detections:
        """Perform inference on a single slice."""
        if self.model is None:
            return sv.Detections.empty()

        results = self.model.predict(
            frame_slice,
            conf=Config.MODEL_CONFIDENCE,
            iou=Config.MODEL_IOU_THRESHOLD,
            device=self.device,
            verbose=False,
            classes=self.selected_class_ids if self.selected_class_ids else None,
        )
        return sv.Detections.from_ultralytics(results[0])

    def get_model_info(self) -> Tuple[str, str]:
        return self.model_name, self.model_type

    def _setup_device(self) -> str:
        """Setup the computation device (GPU/CPU)."""
        if Config.USE_GPU and Config.validate_gpu_availability():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device

    def _load_model(self) -> None:
        """Load the YOLOv11 model, handling both .pt and .engine files."""
        model_path = Config.get_full_model_name()
        self.model_name = os.path.basename(model_path)
        try:
            if model_path.endswith(".engine"):
                self._load_model_with_fallback(model_path, is_tensorrt=True)
            else:
                self._load_model_with_fallback(model_path, is_tensorrt=False)

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _load_model_with_fallback(self, model_path: str, is_tensorrt: bool = False) -> None:
        """Load a model with automatic fallback support."""
        try:
            if is_tensorrt:
                # Handle TensorRT-specific logic
                if not os.path.exists(model_path):
                    logger.warning(f"TensorRT engine not found at {model_path}")
                    logger.info("Attempting to export PyTorch model to TensorRT...")
                    self._export_to_tensorrt(model_path)

                logger.info(f"Loading TensorRT model from {model_path}")
                self.model = YOLO(model_path, task="detect")
                self.model_type = "TensorRT"
            else:
                logger.info(f"Loading PyTorch model from {model_path}")
                self.model = YOLO(model_path)
                self.model.to(self.device)
                self.model_type = "PyTorch"

            # Handle class names for both model types
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
            else:
                # For TensorRT models, try to load class names from corresponding .pt file
                if is_tensorrt:
                    pt_path = os.path.splitext(model_path)[0] + ".pt"
                    if os.path.exists(pt_path):
                        logger.info(f"Loading class names from {pt_path}")
                        temp_model = YOLO(pt_path)
                        self.class_names = temp_model.names
                        del temp_model
                    else:
                        logger.warning(f"Could not find {pt_path} to load class names. Class names will be empty.")
                        self.class_names = {}
                else:
                    self.class_names = {}

            # Filter by selected class names if provided
            if self.selected_class_names:
                all_class_names_map = {v: k for k, v in self.class_names.items()}
                self.selected_class_ids = [
                    all_class_names_map[name]
                    for name in self.selected_class_names
                    if name in all_class_names_map
                ]
                logger.info(f"Filtering for classes: {self.selected_class_names}")
                logger.info(f"Corresponding class IDs: {self.selected_class_ids}")

                # Warn about names not found in the model
                for name in self.selected_class_names:
                    if name not in all_class_names_map:
                        logger.warning(f"Class name '{name}' not found in the model's class list.")

            logger.info(f"{self.model_type} model loaded successfully from {model_path}. Classes: {len(self.class_names)}")

        except Exception as e:
            if is_tensorrt:
                logger.error(f"Failed to load TensorRT model: {e}")
                logger.info("Falling back to regular PyTorch model...")
                self._load_fallback_model(model_path)
            else:
                raise

    def _export_to_tensorrt(self, engine_path: str) -> None:
        """Export PyTorch model to TensorRT format."""
        pt_path = os.path.splitext(engine_path)[0] + ".pt"
        try:
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"PyTorch model not found at {pt_path} for TensorRT export.")

            logger.info(f"Exporting {pt_path} to TensorRT format...")

            model = YOLO(pt_path)

            model.export(
                format="engine",
                half=True,
            )

            logger.info(f"TensorRT export completed successfully for {engine_path}")

        except Exception as e:
            logger.error(f"Failed to export to TensorRT from {pt_path}: {e}")
            raise

    def _load_fallback_model(self, engine_path: str) -> None:
        """Load regular PyTorch model as fallback."""
        pt_path = os.path.splitext(engine_path)[0] + ".pt"
        try:
            if not os.path.exists(pt_path):
                 raise FileNotFoundError(f"PyTorch fallback model not found at {pt_path}.")

            logger.info(f"Loading fallback PyTorch model from {pt_path}")
            self._load_model_with_fallback(pt_path, is_tensorrt=False)
        except Exception as e:
            logger.error(f"Failed to load fallback model from {pt_path}: {e}")
            raise

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Perform object detection on a frame.

        Args:
            frame: Input image frame

        Returns:
            List of detection dictionaries containing bbox, confidence, class info
        """
        if self.model is None:
            return sv.Detections.empty()

        try:
            if Config.USE_SLICER:
                if self.slicer is None:
                    self._init_slicer((frame.shape[0], frame.shape[1]))
                
                assert self.slicer is not None
                # Run inference
                detections = self.slicer(frame)
            else:
                detections = self._perform_inference(frame)

            detections = self.tracker.update_with_detections(detections)
            if self.smoother:
                detections = self.smoother.update_with_detections(detections)

            if Config.USE_POLYGON_ZONE:
                for i, zone in enumerate(self.polygon_zones):
                    mask = zone.trigger(detections=detections)
                    if detections.tracker_id is not None:
                        for tracker_id in detections.tracker_id[mask]:
                            self.zone_tracker_ids[i].add(tracker_id)

            if Config.USE_LINE_ZONE:
                for line_zone in self.line_zones:
                    line_zone.trigger(detections=detections)

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return sv.Detections.empty()

    def draw_detections(self, frame: np.ndarray, detections: sv.Detections,
                       camera_id: int = 0) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Input image frame
            detections: List of detection dictionaries
            camera_id: Camera identifier for color selection

        Returns:
            Frame with drawn detections
        """
        if len(detections) == 0:
            return frame

        labels = []
        if detections.class_id is not None and detections.confidence is not None:
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]

                label_parts = []

                if detections.tracker_id is not None:
                    tracker_id = detections.tracker_id[i]
                    label_parts.append(f"#{tracker_id}")

                label_parts.append(self.class_names.get(class_id, f"ID {class_id}"))
                label_parts.append(f"{confidence:.2f}")

                labels.append(" ".join(label_parts))

        annotated_frame = frame.copy()
        if self.trace_annotator:
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        if self.heat_map_annotator:
            annotated_frame = self.heat_map_annotator.annotate(annotated_frame, detections)

        if Config.USE_POLYGON_ZONE:
            for i, annotator in enumerate(self.polygon_zone_annotators):
                accumulated_count = len(self.zone_tracker_ids[i])
                label = f"Zone {i + 1}: {accumulated_count}"
                annotated_frame = annotator.annotate(scene=annotated_frame, label=label)

        if Config.USE_LINE_ZONE:
            for i, line_zone in enumerate(self.line_zones):
                annotated_frame = self.line_annotators[i].annotate(frame=annotated_frame, line_counter=line_zone)

        if labels:
            annotated_frame = self.label_annotator.annotate(
                annotated_frame, detections, labels
            )

        return annotated_frame
