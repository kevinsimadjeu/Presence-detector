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

├── modelProcessing/

|     ├── model.py                

│     └── processing.py         

│

├── ui.py                    

├── yolov8n.pt            

│

├── README.md      

├── LICENSE   

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

This presence detection program can be coupled to an ESP32 microcontroller for physical actions (e.g. turning on an LED or triggering a relay) depending on the detection of a person.

• Sending the signal to the ESP32 is done via HTTP requests, which makes it easy to reconfigure the IP address or port depending on your network.

• The code is modular: you can adapt the action logic on the ESP32 or add other sensors if necessary.

• This  architecture separates software detection on the computer and hardware control over the ESP32, offering maximum flexibility for different scenarios.



 To start the project, run the ui.py file and it will automatically call the processing file and the  YOLODetector class from the model.py file.


### Warning 

Make sure the ESP32 is connected to the same local network as the computer running the program for communication to work properly.







