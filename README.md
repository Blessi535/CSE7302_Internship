**Object Detection & Measurement System using YOLOv8**
**Overview**
This project is a real-time object detection and measurement system built using YOLOv8, Flask, and OpenCV. It captures live video feed from a camera, detects objects, calculates their dimensions, and estimates weight. The results are streamed and accessed via a mobile browser over a local network.

**System Architecture**
The system follows a client-server architecture:

1) User (Mobile Browser) sends HTTP requests
2) WiFi Network (Same LAN) connects devices
3) Flask Web Server (PC) processes requests
4) OpenCV Camera captures live feed
5) YOLOv8 Model performs object detection
6) Measurement Module calculates size & weight
7) Result Display shows output in browser

**Features
**Real-time object detection using YOLOv8****
1) Live video streaming via Flask
2) Object dimension calculation (Width & Height)
3) Weight estimation (based on size)
4) Mobile browser access via IP
5) Works on local WiFi network (LAN)

**Technologies Used**
Python
Flask (Web Server)
OpenCV (Camera Processing)
YOLOv8 (Object Detection)
NumPy

**Project Structure**
project-folder/
│── app.py                 # Flask application
│── model/                 # YOLOv8 trained model (best.pt)
│── static/                # Images / CSS
│── templates/             # HTML files
│── requirements.txt       # Dependencies
│── README.md              # Documentation

**Installation**

1) Install Dependencies
pip install -r requirements.txt
2) Download YOLOv8 Model
3) Place your trained model:
runs/detect/train/weights/best.pt

**Running the Project**
python app.py
You will get an output like:
Running on http://192.168.x.x:5000
Open this URL in your mobile browser (same WiFi).

**How It Works**
Camera captures live video using OpenCV
Frames are sent to YOLOv8 model
Model detects object using bounding boxes
Measurement module calculates:
Width
Height
Estimated Weight
Flask streams processed video to browser
Results displayed in real-time

**Output**
The system displays:
Object Class
Object Dimensions (Width & Height)
Estimated Weight

**Configuration**
Change camera source in code:
cv2.VideoCapture(0)  # Webcam
For IP camera:
cv2.VideoCapture("http://IP_ADDRESS/video")

**Limitations**
Accuracy depends on trained dataset
Weight estimation is approximate
Requires good lighting conditions
Works only within same network (LAN)

**Future Enhancements**
Add multiple object detection
Improve weight estimation using ML regression
Deploy on cloud
Add database storage
Integrate Raspberry Pi for portability

**Use Cases**
Fruit/Vegetable quality inspection
Smart agriculture
Industrial object measurement
Inventory systems
