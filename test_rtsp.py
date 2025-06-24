import os
import cv2
import argparse  # New import
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv(override=True)
# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Test RTSP stream with specified backend.")
parser.add_argument(
    "--backend",
    type=str,
    default="GStreamer",  # Default to GStreamer
    choices=["FFMPEG", "GStreamer"],
    help="Specify the backend for VideoCapture: FFMPEG or GStreamer"
)
args = parser.parse_args()

# --- Backend Specific Configuration ---
rtsp_url = os.getenv("CAMERA_STREAMS", "rtsp://admin:admin")
cap = None

if args.backend == "FFMPEG":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;timeout;5000"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
elif args.backend == "GStreamer":
    # Construct the GStreamer pipeline string using decodebin for automatic element selection
    gst_pipeline = (
        f"rtspsrc location=\"{rtsp_url}\" protocols=tcp ! "
        "decodebin ! videoconvert ! appsink drop=1"
    )
    print(f"Using GStreamer pipeline: {gst_pipeline}")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if cap is None or not cap.isOpened():
    print(f"Error: Could not open RTSP stream with {args.backend} backend.")
    exit()

print(f"Using {args.backend} backend.")
print("RTSP Stream URL:", rtsp_url)  # cap.getBackendName() might not be reliable for GStreamer with custom pipelines
print("Is VideoCapture opened:", cap.isOpened())
# get stream properties
print("Frame Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
