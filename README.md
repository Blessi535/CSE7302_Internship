# Object Detection & Measurement System using YOLOv8

## 📌 Overview
This project is a **real-time object detection and measurement system** built using **YOLOv8, Flask, and OpenCV**.  
It captures live video from a camera, detects objects, calculates their dimensions, and estimates weight.  

The results are streamed and accessed via a **mobile browser over a local network (LAN)**.

---

##  System Architecture
The system follows a **client-server architecture**:

1. User (Mobile Browser) sends HTTP requests  
2. WiFi Network (Same LAN) connects devices  
3. Flask Web Server (PC) processes requests  
4. OpenCV Camera captures live feed  
5. YOLOv8 Model performs object detection  
6. Measurement Module calculates size & weight  
7. Result Display shows output in browser  

---

## ✨ Features
- Real-time object detection using YOLOv8  
- Live video streaming via Flask  
- Object dimension calculation (Width & Height)  
- Weight estimation (based on size)  
- Mobile browser access via IP  
- Works on local WiFi network (LAN)  

---

## 🛠️ Technologies Used
- Python  
- Flask (Web Server)  
- OpenCV (Camera Processing)  
- YOLOv8 (Object Detection)  
- NumPy  

---

## 📁 Project Structureproject-folder/
│── app.py # Flask application

│── model/ # YOLOv8 trained model (best.pt)

│── static/ # Images / CSS

│── templates/ # HTML files

│── requirements.txt # Dependencies

│── README.md # Documentation


---

## ⚙️ Installation

### 1️⃣ Install Dependencies
pip install -r requirements.txt


## 2️⃣ Download YOLOv8 Model

Train the model with 50 epochs time taken is 30 to 40 minutes.

## 3️⃣ Place your trained model:

runs/detect/train/weights/best.pt

▶️ Running the Project
python app.py

You will get an output like:

Running on http://192.168.x.x:5000

Open this URL in your mobile browser (same WiFi network).

---
## 🔍 How It Works

Camera captures live video using OpenCV

Frames are sent to YOLOv8 model

Model detects objects using bounding boxes

Measurement module calculates:

-Width

-Height

-Estimated Weight

-Flask streams processed video to browser

Results are displayed in real-time

---
## 📊 Output

The system displays:

1) Object Class
2) Object Dimensions (Width & Height)
3) Estimated Weight
---
## ⚙️ Configuration

Change Camera Source:

cv2.VideoCapture(0)  # Webcam

For IP Camera:

cv2.VideoCapture("http://IP_ADDRESS/video")

---
## ⚠️ Limitations

1) Accuracy depends on trained dataset
2) Weight estimation is approximate
3) Requires good lighting conditions
4) Works only within the same network (LAN)
---
## 💡 Use Cases

Fruit/Vegetable quality inspection

Smart agriculture

Industrial object measurement

Inventory systems
