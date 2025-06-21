"""
Real-time object detection from a YouTube video stream.
"""
import cv2
import time
import signal
import sys
import numpy as np
from loguru import logger
import yt_dlp

from config import Config
from detector.object_detector import ObjectDetector

# Setup logging
# logger.add("logs/youtube_app.log", rotation="10 MB", retention="7 days", level="INFO")

class YouTubeObjectDetection:
    """Main class for YouTube object detection."""

    def __init__(self):
        """Initialize the detection system."""
        self.detector = None
        self.video_capture = None
        self.running = False
        self.stats = {
            'total_frames': 0,
            'detection_time': 0.0,
            'display_time': 0.0
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()
    
    def _get_1080p_stream_url(self, youtube_url) -> str:
        ydl_opts = {
            'quiet': True,
            'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[height<=1080]',
            'merge_output_format': 'mp4',
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            formats = info_dict.get('formats', [info_dict])
            
            # Find the best video stream with height = 1080
            for f in formats:
                if f.get('height') == 1080 and f.get('vcodec') != 'none':
                    return f['url']
            
            # Fallback: return the best available
            logger.warning("1080p stream not found, returning best available stream.")
            return info_dict['url']
        
    def _get_stream_url(self, youtube_url: str) -> str | None:
        """
        Get the stream URL for a given YouTube URL.
        Tries to get the best 1080p mp4 stream, falls back to best available.
        """
        logger.info(f"Getting stream URL for {youtube_url}")
        
        ydl_opts = {
            'quiet': True,
            'format': 'bestvideo[ext=mp4][height<=1080][vcodec=h264]+bestaudio[ext=m4a]/best[ext=mp4][vcodec=h264]/best',
            'noplaylist': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)
                
                if not info_dict:
                    logger.error("yt-dlp failed to extract info.")
                    return None
                
                # if 'url' in info_dict:
                #     return info_dict['url']
                
                formats = info_dict.get('formats', [info_dict])
                for f in reversed(formats):
                    if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('url') and f.get('height') == 1080:
                        logger.info(f"Found mp4 stream: {f.get('format_note')}")
                        return f['url']
                
                logger.warning("No direct mp4 stream found. Falling back to the first available stream URL.")
                if formats and formats[0].get('url'):
                    return formats[0]['url']
                
                logger.error("Could not find any stream URL.")
                return None

        except Exception as e:
            logger.error(f"yt-dlp failed to get stream URL: {e}")
            return None

    def initialize(self) -> bool:
        """Initialize the detection system components."""
        try:
            # Initialize object detector
            logger.info("Initializing object detector...")
            self.detector = ObjectDetector()

            # Get YouTube stream URL
            
            youtube_url = Config.YOUTUBE_URL
            if not youtube_url:
                logger.error("YOUTUBE_URL not set in config or .env file. Exiting.")
                return False            
            
            logger.info(f"Getting YouTube video stream from {youtube_url}")
            stream_url = self._get_stream_url(youtube_url)
            if not stream_url:
                logger.error("Failed to get stream URL. Exiting.")
                return False

            logger.info(f"Streaming from: {stream_url}")
            self.video_capture = cv2.VideoCapture(stream_url)

            if not self.video_capture.isOpened():
                logger.error("Failed to open YouTube stream. Exiting.")
                return False

            logger.info("System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with object detection.

        Args:
            frame: Input frame

        Returns:
            Processed frame with detections
        """
        try:
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))

            # Perform object detection
            start_time = time.time()
            detections = self.detector.detect(frame)
            detection_time = time.time() - start_time

            # Draw detections
            frame_with_detections = self.detector.draw_detections(frame, detections, 0) # camera_id=0
            model_name, model_type = self.detector.get_model_info()

            # Add FPS and info
            if Config.DISPLAY_FPS:
                # A simple FPS calculation
                self.stats['total_frames'] += 1
                self.stats['detection_time'] += detection_time
                avg_detection_time = self.stats['detection_time'] / self.stats['total_frames']
                fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
                info_text = f"FPS: {fps:.1f} | Objects: {len(detections)}"
                cv2.putText(
                    frame_with_detections, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    frame_with_detections, f"Model: {model_name} ({model_type})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

            return frame_with_detections

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def run(self):
        """Run the main detection loop."""
        if not self.initialize():
            return

        self.running = True
        logger.info("Starting YouTube object detection system...")

        # Calculate delay for playback FPS
        playback_delay = int(1000 / Config.PLAYBACK_FPS)

        while self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                logger.warning("End of stream or cannot read frame.")
                break

            processed_frame = self.process_frame(frame)

            # Display the frame
            cv2.imshow("YouTube Object Detection", processed_frame)

            if cv2.waitKey(playback_delay) & 0xFF == ord('q'):
                logger.info("'q' pressed. Stopping system.")
                break

        self.stop()

    def stop(self):
        """Stop the system and release resources."""
        self.running = False
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        logger.info("System stopped.")

if __name__ == "__main__":
    # To run this, make sure you have a YOUTUBE_URL in your .env file
    # e.g., YOUTUBE_URL="https://www.youtube.com/watch?v=..."
    # You also need to install yt-dlp: pip install yt-dlp
    system = YouTubeObjectDetection()
    system.run()
