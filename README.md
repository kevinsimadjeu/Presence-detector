# Presence-detector

A real-time presence detection system capable of identifying people and various objects using Computer Vision and the YOLO (You Only Look Once) deep learning model.

This project demonstrates how to build an efficient detection pipeline, process camera frames, and generate bounding boxes with confidence scores.



# FEATURER

Real-time detection using webcam or video input

Supports multiple class (persons,animals, objects, etc.)

Uses YOLOv4 / YOLOv5 / YOLOv8 (depending on your model choice)

Fast and accurate inference

Modular and easy to extend

Lightweight UI  of detections



 # MAIN TECHNOLOGIES AND MODULES USED
 
 • Python 3
 
 • OpenCV
 
 • YOLO
 
 • NumPy
 
 • Tkinter



# PROJECT STRUCTURE


project-root/

│

├── YOLO/

│   └── yolov8.py 
│

├── modelProcessing/

│ 

|   ├── model.py               # Model management (loading, configuration)

│   └── processing.py          # Data preprocessing and post-processing

│

├── ui.py                      #(GUI)

├── yolov8n.pt                 # Pre-trained YOLOv8 weights

│

├── README.md                  # Project overview, installation, usage

├── LICENSE                    # Project license

└── .gitattributes             # Git configuration



# MODEL FILES

Because YOLO weight files are very large (>200 MB), they are tracked using Git LFS.

If needed, install Git LFS:

``` git lfs install ```

Then download the model:

``` git lfs pull ```


# Installation

Clone the repository:

 ```
git clone https://github.com/kevinsimadjeu/Presence-detector.git. 

```


# INSTRUCTION: 


 To start the project, run the ui.py file and it will automatically call the processing file and the  YOLODetector class from the model.py file.







