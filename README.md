# MemryX_on_RPi5</br>

## Scripts:
  - infer_dfp_videos_letterboxed_v8m.py: Contianer script to run inference on MemryX using yolo v8m
   - yolov8m_letterboxed.py: Edited yolo script to letterbox input images</br> 

  - infer_dfp_videos_letterboxed_v8s.py: Contianer script to run inference on MemryX using yolo v8s
  - yolov8s_letterboxed.py: Edited yolo script to letterbox input images</br> 

## Model Files:
   - dfp and post tflite files for both models downloaded from MemryX model explorer</br>

## Misc Info:
  - running on a 1080p webcam, but passing 800x600 to pipeline
  - There is a naming convention oddity in the yolov8s container script; "from yolov8s_letterboxed import yolov8m as YoloModel"
    because we reused the yolo file, the name wasn't changed from yolov8m to yolov8s. It is just the class name, so it shouldn't
   change functionality.</br> 

## Problem:
  - The yolov8s version of the script runs fine indefinitely, the yolov8m version will freeze after ~5 minutes. 
  - Our hypothesis is that there are issues with the queue filling up, but we are not familiar enough to troubleshoot.
