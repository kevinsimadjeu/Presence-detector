# processing.py
import cv2
import time
from datetime import datetime
from collections import deque
from modelProcessing.model import YOLODetector 

#Classe de traitement lors des captures videos
class Processing:
    def __init__(self):
        self.detector = YOLODetector()
        self.confidence_threshold = 0.5
        self.running = False
        self.cap = None
        self.recording = False
        self.video_writer = None
        self.current_frame = None

        self.stats = {
            'total_detections': 0,
            'fps': 0,
            'objects_per_frame': deque(maxlen=100),
            'detection_history': {},
            'start_time': None
        }

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.stats['start_time'] = time.time()
        self.stats['total_detections'] = 0
        self.stats['detection_history'] = {}

        return True

    def stop(self):
        self.running = False
        if self.recording:
            self.toggle_recording()
        if self.cap:
            self.cap.release()

    def toggle_recording(self):
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'detection_{timestamp}.avi'
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                self.recording = True
        else:
            if self.video_writer:
                self.video_writer.release()
            self.recording = False

    def screenshot(self):
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'screenshot_{timestamp}.jpg'
            cv2.imwrite(filename, self.current_frame)
            return filename

    def loop(self):
        fps_start_time = time.time()
        fps_counter = 0

        ret, frame = self.cap.read()
        if not ret:
            return None

        results = self.detector.predict(frame)
        num_detections = len(results.boxes)

        if num_detections > 0:
            self.stats['total_detections'] += num_detections
            self.stats['objects_per_frame'].append(num_detections)
            for box in results.boxes:
                class_id = int(box.cls[0])
                label = results.names[class_id]
                self.stats['detection_history'][label] = \
                    self.stats['detection_history'].get(label, 0) + 1

        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            self.stats['fps'] = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        annotated_frame = results.plot()

        if self.recording and self.video_writer:
            self.video_writer.write(annotated_frame)

        self.current_frame = annotated_frame.copy()
        return annotated_frame
