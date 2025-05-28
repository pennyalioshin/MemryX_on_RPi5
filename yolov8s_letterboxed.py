"""
============
Information:
============
Project: yolov8m object detection on RPI5 using MemryX AI accelerator
File Name: yolov8m_letterboxed.py

============
Description:
============
Script that passes all of the yolo functions to the "header" file "infer_dfp_videos_letterboxed."
This is where all the image processing and detection takes place. 
"""

###################################################################################################

# Imports

import numpy as np
import cv2

###################################################################################################

COCO_CLASSES = ( "person", "bicycle", "car", "motorcycle", "airplane", "bus",
        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
        "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",)

###################################################################################################
###################################################################################################
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    """
    Resize and pad image while meeting stride-multiple constraints
    """
    shape = img.shape[:2]  # (h, w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, dw, dh
###################################################################################################

class yolov8m:
    """
    A helper class to run YOLOv8 pre- and post-proccessing.
    """

###################################################################################################
    def __init__(self, stream_img_size=None):
        """
        The initialization function.
        """

        self.name = 'yolov8m on MXA'
        #self.input_size = (640,640,3) 
        self.input_width = 640
        self.input_height = 640
        self.confidence_thres = 0.7
        self.iou_thres = 0.3

        if stream_img_size:
            self.stream_mode = True
            dummy = np.zeros(stream_img_size, dtype=np.uint8)
            # Pre-calculate ratio/pad values for preprocessing
            self.preprocess(np.zeros(stream_img_size))
        else:
            self.stream_mode = False

###################################################################################################
    def preprocess(self, img):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        self.input_img = img

        # Get the height and width of the input image
        self.img_height, self.img_width = img.shape[:2]
        #print(f"Input {self.input_width}, {self.input_height} <-> {self.img_width, self.img_height}")

        img, self.ratio, self.dw, self.dh = letterbox(img, new_shape=(self.input_height, self.input_width))
        
        img = img.astype(np.float32) / 255.0
        image_data = np.expand_dims(img, axis=2)

        # Return the preprocessed image data
        return image_data


###################################################################################################
    def postprocess(self, output):
        """
        Performs post-processing on the YOLOv8 model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            list: A list of detections where each detection is a dictionary containing 
                    'bbox', 'class_id', 'class', and 'score'.
        """
        # Transpose the output to shape (8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))

        boxes = outputs[:, :4]  # x_center, y_center, w, h
        class_scores = outputs[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        valid_indices = np.where(max_scores >= self.confidence_thres)[0]
        if len(valid_indices) == 0:
            return []

        boxes = boxes[valid_indices]
        class_ids = class_ids[valid_indices]
        scores = max_scores[valid_indices]

        # Convert xywh to xyxy
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2

        # Undo letterbox padding
        boxes[:, [0, 2]] -= self.dw
        boxes[:, [1, 3]] -= self.dh
        boxes /= self.ratio

        boxes[:, 0] = boxes[:, 0].clip(0, self.img_width)
        boxes[:, 1] = boxes[:, 1].clip(0, self.img_height)
        boxes[:, 2] = boxes[:, 2].clip(0, self.img_width)
        boxes[:, 3] = boxes[:, 3].clip(0, self.img_height)

        detections = [{
            'bbox': [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2] - boxes[i][0]), int(boxes[i][3] - boxes[i][1])],
            'class_id': int(class_ids[i]),
            'class': COCO_CLASSES[int(class_ids[i])],
            'score': scores[i]
        } for i in range(len(boxes))]

        if detections:
            boxes_for_nms = [d['bbox'] for d in detections]
            scores_for_nms = [d['score'] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, self.confidence_thres, self.iou_thres)

            if len(indices) > 0:
                if isinstance(indices[0], (list, np.ndarray)):
                    indices = [i[0] for i in indices]
                detections = [detections[i] for i in indices]
            else:
                detections = []

        return detections


###################################################################################################
if __name__=="__main__":
    pass

# eof
