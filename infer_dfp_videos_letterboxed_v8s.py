"""
============
Information:
============
Project: yolov8m object detection on RPI5 using MemryX AI accelerator
File Name: infer_dfp_videos_letterboxed.py

============
Description:
============
A script designed to run yolov8m inference on a live webcam feed using the MemryX M.2 as a hardware
accelerator. Built off of example code given on the MemryX developper hub.

============
Notes:
============
This file is the "header" file. In order for this code to run properly, it must be run from the same
directory as "yolov8m_letterboxed" as well as in an environment with tensorflow and all other necessary
libraries installed.

***THE ONLY TWO LINES YOU NEED TO CHANGE TO MAKE THIS CODE WORK ON YOUR DEVICE ARE LINES 99 AND 100. 
    REPLACE THE PATH AFTER "dfp=" WITH THE PATH TO YOUR dfp, AND REPLACE THE PATH AFTER
    set_postprocessing_model WITH THE PATH TO YOUR xxxxx_post.tflite****
"""

###################################################################################################

# Imports
import time
import argparse
import numpy as np
import cv2
from queue import Queue, Full
from threading import Thread
from memryx import MultiStreamAsyncAccl
from yolov8s_letterboxed import yolov8m as YoloModel

###################################################################################################
###################################################################################################
###################################################################################################

class yolov8m_MXA:
    """
    A demo app to run yolov8m on the MemryX MXA.
    """

###################################################################################################
    def __init__(self, video_paths, show=True):
        """
        Initialization function.
        """

        # Controls
        self.show = show
        self.done = False
        self.num_streams = len(video_paths)

        # Stream-related containers
        self.streams = []
        self.streams_idx = [True] * self.num_streams
        self.stream_window = [False] * self.num_streams
        self.cap_queue = {i: Queue(maxsize=10) for i in range(self.num_streams)}
        self.dets_queue = {i: Queue(maxsize=10) for i in range(self.num_streams)}
        self.outputs = {i: [] for i in range(self.num_streams)}
        self.dims = {}
        self.color_wheel = {}
        self.model = {}

        # Timing and FPS
        self.dt_index = {i: 0 for i in range(self.num_streams)}
        self.frame_end_time = {i: 0 for i in range(self.num_streams)}
        self.fps = {i: 0 for i in range(self.num_streams)}
        self.dt_array = {i: np.zeros(30) for i in range(self.num_streams)}
        self.writer = {i: None for i in range(self.num_streams)}

        # Initialize video captures, models, and dimensions for each streams
        for i, video_path in enumerate(video_paths):
            vidcap = cv2.VideoCapture(video_path)
            self.streams.append(vidcap)

            vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

            self.dims[i] = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.color_wheel[i] = np.random.randint(0, 255, (20, 3)).astype(np.int32)
            print(f"video source {i} frame dims: {self.dims[i][0]} x {self.dims[i][1]}")

            # Initialize the model with set stream dimensions
            self.model[i] = YoloModel(stream_img_size=(self.dims[i][1], self.dims[i][0], 3))
        
        self.display_thread = Thread(target=self.display)

###################################################################################################
    def run(self):
        """
        Function responsible for starting inference on the MXA.
        """
        accl = MultiStreamAsyncAccl(dfp='/home/palioshin/BDTI_VLM/yolov8s_on_MXA/yolov8s.dfp')
        accl.set_postprocessing_model('/home/palioshin/BDTI_VLM/yolov8s_on_MXA/yolov8s_post.tflite', model_idx=0)

        self.display_thread.start()

        # Connect the input and output functions and let the accl run
        accl.connect_streams(self.capture_and_preprocess, self.postprocess, self.num_streams)
        accl.wait()

        self.done = True

        # Join the display thread
        self.display_thread.join()

###################################################################################################
    def capture_and_preprocess(self, stream_idx):
        """
        Captures a frame for the video device and pre-processes it.
        """
        got_frame, frame = self.streams[stream_idx].read()

        if not got_frame:
            self.streams_idx[stream_idx] = False
            return None

        try:
            # Put the frame in the cap_queue to be processed
            self.cap_queue[stream_idx].put(frame, timeout=2)

            # Pre-process the frame using the corresponding model
            frame = self.model[stream_idx].preprocess(frame)
            return frame

        except Full:
            print('Dropped frame .. exiting')
            return None

###################################################################################################
    def postprocess(self, stream_idx, *mxa_output):
        """
        Post-process the MXA output.
        """
        dets = self.model[stream_idx].postprocess(mxa_output)

        # Push the detection results to the queue
        self.dets_queue[stream_idx].put(dets)

        # Calculate the FPS
        self.dt_array[stream_idx][self.dt_index[stream_idx]] = time.time() - self.frame_end_time[stream_idx]
        self.dt_index[stream_idx] += 1

        if self.dt_index[stream_idx] % 15 == 0:
            self.fps[stream_idx] = 1 / np.average(self.dt_array[stream_idx])

        if self.dt_index[stream_idx] >= 30:
            self.dt_index[stream_idx] = 0

        self.frame_end_time[stream_idx] = time.time()

###################################################################################################
    def display(self):
        """
        Continuously draws bounding boxes over the original image. Refresehes for each frame.
        """
        while not self.done:
            # Iterate over each stream to handle multiple displays
            for stream_idx in range(self.num_streams):
                # Check if the queues for frames and detections have data
                if not self.cap_queue[stream_idx].empty() and not self.dets_queue[stream_idx].empty():
                    frame = self.cap_queue[stream_idx].get()
                    dets = self.dets_queue[stream_idx].get()

                    # Draw detection boxes on the frame
                    for d in dets:

                        x1, y1, w, h = d['bbox']
                        
                        color = tuple(int(c) for c in self.color_wheel[stream_idx][d['class_id'] % 20])

                        # Draw the bounding box on the image
                        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
                        # Draw labels on the bounding boxes
                        frame = cv2.putText(frame, f"{d['class']} {d['score']:2.1f}", (x1 + 2, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


                    # Add FPS information to the frame
                    fps_text = f"{self.model[stream_idx].name} - {self.fps[stream_idx]:.1f} FPS" if self.fps[stream_idx] > 1 else self.model[stream_idx].name
                    frame = cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Show the frame in a unique window for each stream
                    if self.show:
                        window_name = f"Stream {stream_idx} - YOLOv8m"
                        cv2.imshow(window_name, frame)

            # Exit on key press (applies to all streams)
            if cv2.waitKey(1) == ord('q'):
                self.done = True

        # When done, destroy all windows and release resources
        cv2.destroyAllWindows()
        for stream in self.streams:
            stream.release()

###################################################################################################
###################################################################################################
###################################################################################################

def main(args):
    """
    The main funtion
    """

    yolo8m_inf = yolov8m_MXA(video_paths = args.video_paths, show=args.show)
    yolo8m_inf.run()

###################################################################################################

if __name__=="__main__":
    # The args parser
    parser = argparse.ArgumentParser(description = "\033[34mMemryX Yolo Demo\033[0m")
    parser.add_argument('--video_paths', nargs='+',  dest="video_paths", 
                        action="store", 
                        default=['/dev/video0'],
                        help="the path to video file to run inference on. Use '/dev/video0' for a webcam. (Default:'/dev/video0')")
    parser.add_argument('--no_display', dest="show", 
                        action="store_false", 
                        default=True,
                        help="Optionally turn off the video display")

    args = parser.parse_args()

    # Call the main function
    main(args)

# eof
