# Presence-detector

A real-time presence detection system capable of identifying people and various objects using Computer Vision and the YOLO (You Only Look Once) deep learning model.

This project demonstrates how to build an efficient detection pipeline, process camera frames, and generate bounding boxes with confidence scores.



#FEATURER

Real-time detection using webcam or video input

Supports multiple classes (persons, objects, etc.)

Uses YOLOv4 / YOLOv5 / YOLOv8 (depending on your model choice)

Fast and accurate inference

Modular and easy to extend

Lightweight UI  of detections



 #Technologies and modules used 
- Python 3
- OpenCV
- YOLO 
- NumPy
- Tkinter



#PROJECT STRUCTURE


----> ReadMe.md

  ── models/
  
    • yolov4.weights   # Large files stored via Git LFS
  
    • yolov8n.pt
  
  ── src/
  
    • ui.py
 
    • model.py

 

#MODEL FILES

Because YOLO weight files are very large (>200 MB), they are tracked using Git LFS.

If needed, install Git LFS:

" git lfs install "

Then download the model:

" git lfs pull "



INSTRUCTION:  To start the project, run the ui.py file and it will automatically call the YOLODetector class from the model.py file.






