"""
Real-time instance segmentation from a YouTube video stream using Ultralytics Solutions.
"""
import cv2
import time
import signal
import sys
import numpy as np
from loguru import logger
import yt_dlp
from ultralytics import solutions
from utils import get_stream_url
from config import Config

# Setup logging
# logger.add("logs/youtube_segmentation_app.log", rotation="10 MB", retention="7 days", level="INFO")

class YouTubeSegmentationSystem:
    """Main class for YouTube instance segmentation."""

    def __init__(self):
        """Initialize the segmentation system."""
        self.isegment = None
        self.video_capture = None
        self.running = False
        self.stats = {
            'total_frames': 0,
            'segmentation_time': 0.0,
            'display_time': 0.0
        }

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received. Stopping system...")
        self.stop()

    # def _get_stream_url(self, youtube_url: str) -> str | None:
    #     """
    #     Get the stream URL for a given YouTube URL.
    #     Tries to get the best 1080p mp4 stream, falls back to best available.
    #     """
    #     logger.info(f"Getting stream URL for {youtube_url}")
        
    #     ydl_opts = {
    #         'quiet': True,
    #         'format': 'bestvideo[ext=mp4][height<=1080][vcodec=h264]+bestaudio[ext=m4a]/best[ext=mp4][vcodec=h264]/best',
    #         'noplaylist': True,
    #     }

    #     try:
    #         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #             info_dict = ydl.extract_info(youtube_url, download=False)
                
    #             if not info_dict:
    #                 logger.error("yt-dlp failed to extract info.")
    #                 return None               
                
    #             formats = info_dict.get('formats', [info_dict])
    #             for f in reversed(formats):
    #                 if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('url') and f.get('height') == 1080:
    #                     logger.info(f"Found mp4 stream: {f.get('format_note')}")
    #                     return f['url']
                
    #             logger.warning("No direct mp4 stream found. Falling back to the first available stream URL.")
    #             if formats and formats[0].get('url'):
    #                 return formats[0]['url']
                
    #             logger.error("Could not find any stream URL.")
    #             return None

    #     except Exception as e:
    #         logger.error(f"yt-dlp failed to get stream URL: {e}")
    #         return None

    def initialize(self) -> bool:
        """Initialize the segmentation system components."""
        try:
            # Initialize instance segmentation using Ultralytics Solutions
            logger.info("Initializing instance segmentation with Ultralytics Solutions...")
            model_path = Config.get_full_segmentation_model_name()
            
            self.isegment = solutions.InstanceSegmentation(
                show=False,  # We'll handle display ourselves
                model=model_path,
                conf=Config.MODEL_CONFIDENCE,
                iou=Config.MODEL_IOU_THRESHOLD,
                verbose=False,
            )

            # Get YouTube stream URL
            youtube_url = Config.YOUTUBE_URL
            if not youtube_url:
                logger.error("YOUTUBE_URL not set in config or .env file. Exiting.")
                return False            
            
            logger.info(f"Getting YouTube video stream from {youtube_url}")
            stream_url = get_stream_url(youtube_url)

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
        Process a single frame with instance segmentation.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with segmentations
        """
        try:
            # Resize frame if necessary
            if frame.shape[1] != Config.WINDOW_WIDTH or frame.shape[0] != Config.WINDOW_HEIGHT:
                frame = cv2.resize(frame, (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
            
            if self.isegment is None:
                logger.error("Instance segmentation not initialized")
                return frame
            
            start_time = time.time()
            results = self.isegment(frame)
            segmentation_time = time.time() - start_time
            
            if results is not None and hasattr(results, 'plot_im') and results.plot_im is not None:
                processed_frame = results.plot_im
            else:
                processed_frame = frame
            
            if Config.DISPLAY_FPS:
                self.stats['total_frames'] += 1
                self.stats['segmentation_time'] += segmentation_time
                avg_segmentation_time = self.stats['segmentation_time'] / self.stats['total_frames']
                fps = 1.0 / avg_segmentation_time if avg_segmentation_time > 0 else 0
                info_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    processed_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                time_text = f"Segmentation: {segmentation_time*1000:.1f}ms"
                cv2.putText(
                    processed_frame, time_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def run(self):
        """Run the main segmentation loop."""
        if not self.initialize():
            return

        self.running = True
        logger.info("Starting YouTube instance segmentation system...")

        playback_delay = int(1000 / Config.PLAYBACK_FPS)
        window_name = "YouTube Instance Segmentation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)

        try:
            while self.running:
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.warning("End of stream or cannot read frame.")
                    break

                start_time = time.time()
                processed_frame = self.process_frame(frame)
                
                cv2.imshow(window_name, processed_frame)
                
                display_time = time.time() - start_time
                self.stats['display_time'] += display_time

                # key = cv2.waitKey(playback_delay) & 0xFF
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("'q' or ESC pressed. Stopping system.")
                    break
                elif key == ord('s'):
                    self._save_screenshot(processed_frame)
                elif key == ord('r'):
                    self._reset_stats()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save a screenshot of the current frame."""
        timestamp = int(time.time())
        filename = f"youtube_segmentation_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")

    def _reset_stats(self) -> None:
        """Reset performance statistics."""
        self.stats = {
            'total_frames': 0,
            'segmentation_time': 0.0,
            'display_time': 0.0
        }
        logger.info("Statistics reset")

    def stop(self):
        """Stop the system and release resources."""
        self.running = False
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()

        if self.stats['total_frames'] > 0:
            avg_segmentation_time = self.stats['segmentation_time'] / self.stats['total_frames']
            avg_display_time = self.stats['display_time'] / self.stats['total_frames']
            
            logger.info("=== Final Statistics ===")
            logger.info(f"Total frames processed: {self.stats['total_frames']}")
            logger.info(f"Average segmentation time: {avg_segmentation_time*1000:.2f}ms")
            logger.info(f"Average display time: {avg_display_time*1000:.2f}ms")

        logger.info("System stopped.")

def main():
    """Main function."""
    print("ðŸŽ¯ Real-time YouTube Instance Segmentation System v1.0")
    print("======================================================")
    print("Features:")
    print("- Real-time YOLOv11 instance segmentation from YouTube")
    print("- GPU acceleration (if available)")
    print("- Advanced visualization")
    print()
    print("Controls:")
    print("- Press 'q' or ESC to quit")
    print("- Press 's' to save a screenshot")
    print("- Press 'r' to reset statistics")
    print()
    
    system = YouTubeSegmentationSystem()
    try:
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # To run this, make sure you have a YOUTUBE_URL in your .env file
    # e.g., YOUTUBE_URL="https://www.youtube.com/watch?v=..."
    # You also need to install yt-dlp: pip install yt-dlp
    main()
