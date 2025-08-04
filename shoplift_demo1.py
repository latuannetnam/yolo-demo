# this is a demo of using the roboflow inference pipeline to run a YOLOv8 model on a video file
# reference: https://universe.roboflow.com/yolo-merge/shoplifting-x4fxj/model/2
# import the InferencePipeline interface
from inference import InferencePipeline
# import a built in sink called render_boxes (sinks are the logic that happens after inference)
from inference.core.interfaces.stream.sinks import render_boxes
import os
import dotenv
# load the environment variables from the .env file
dotenv.load_dotenv(override=True)
ROBOFLOW_API_KEY=os.getenv("ROBOFLOW_API_KEY")
print(f"ROBOFLOW_API_KEY: {ROBOFLOW_API_KEY}")
# create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="shoplifting-x4fxj/2", # set the model id to a yolov8x model with in put size 1280
    # set the video reference (source of video), it can be a link/path to a video file, an RTSP stream url, or an integer representing a device id (usually 0 for built in webcams)
    video_reference="rtsp://admin:admin123@117.0.0.18:5551/cam/realmonitor?channel=1&subtype=0", 
    on_prediction=render_boxes, # tell the pipeline object what to do with each set of inference by passing a function
    api_key=os.getenv("ROBOFLOW_API_KEY"), # provide your roboflow api key for loading models from the roboflow api
)
# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()