"""
    Real-Time Tracking with YOLOv5 
    Copyright (C) 2022 Felix Lu fei123ilike@gmail.com

"""
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from sort import Sort
import argparse

def get_color(number):
    """ Converts an integer number to a color """
    # change these however you want to
    blue = int(number*30 % 256)
    green = int(number*103 % 256)
    red = int(number*50 % 256)

    return (red, blue, green)

class ObjectTracking:

    def __init__(self, args):
       
        self.source = args.source
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        self.mot_tracker = self.load_tracker(args)
        
        self.CLASS_NAMES_DICT = self.model.model.names

        self.CLASS_IDS = [x for x in range(len(self.CLASS_NAMES_DICT))]

    def load_tracker(self, args):

        """create instance of the SORT tracker"""

        return Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) 
        
    def load_model(self):

        """load a pretrained YOLO model"""
       
        model = YOLO("yolov8m.pt")  
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        xyxys = []
        confidences = []
        class_ids = []
        dets = []
        results = results[0]
        
        # Extract detections 
        for i in range(len(results.boxes)):
            boxes = results.boxes.cpu().numpy()[i]
            class_id = boxes.cls.astype(int)
            conf = boxes.conf
            xyxy = boxes.xyxy.astype(int).tolist()[0]

            if class_id in self.CLASS_IDS:
                xyxys.append(xyxy)
                confidences.append(*conf)
                class_ids.append(*class_id)
                dets.append([*xyxy, *conf, *class_id])

        return frame, dets, xyxys, confidences, class_ids
    
    
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        if self.source.isdigit(): 
            self.source = int(self.source)
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        colours = (np.random.rand(32, 3) * 255).astype(int) # random color for display bbox
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)

            # detections --> N X (x, y, x, y, conf, cls)
            frame, dets, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
            dets = np.array(dets)[:,:4]

            # tracker--> (x, y, x, y, conf)
            trackers = self.mot_tracker.update(dets)
            tracking_ids = trackers[:, -1].tolist()
            
            for xyxy, conf, cls, id in zip(xyxys, confidences, class_ids, tracking_ids):
                label = f"{self.CLASS_NAMES_DICT[cls]} {id} Conf: {conf:0.2f}"
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), get_color(cls), 1)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(cls), 2)

            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow('Tracking', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Real-Time Tracking demo')
    parser.add_argument('--source', help="input source either webcam or video file.", type=str, default='0')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=5)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=0)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':  

    args = parse_args()

    object_tracker = ObjectTracking(args = args)
    object_tracker()