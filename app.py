import cv2
from ultralytics import YOLO
from collections import deque
import datetime
import os
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
MODEL_PATH = "runs/detect/onion_model/weights/best.pt"
CONF_THRESHOLD = 0.6
HISTORY_LEN = 10
REPORT_DIR = "onion_reports"
SNAPSHOT_DIR = "onion_snapshots"

# Mapping from your data.yaml (Ensure indices match your YAML file)
CLASS_NAMES = {
    0: "Indian Sambar Onion",
    1: "Light Yellow Onion",
    2: "Red Onion",
    3: "Spring Onion",
    4: "White Onion",
    5: "Yellow Onion"
}

# Initialize Folders
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Load Model
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
history = deque(maxlen=HISTORY_LEN)
current_report_file = os.path.join(REPORT_DIR, f"report_{datetime.date.today()}.csv")

def estimate_weight(w, h, onion_type):
    # Adjust multiplier based on onion density/type if needed
    multiplier = 0.0025 if "Sambar" in onion_type else 0.002
    return round((w * h) * multiplier, 1)

def check_freshness(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return "Unknown", 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    # Freshness logic: Very dark or very dull spots usually indicate rot
    status = "Fresh" if avg_brightness > 110 else "Suspect"
    return status, int(avg_brightness)

def log_to_csv(data):
    df = pd.DataFrame([data])
    file_exists = os.path.isfile(current_report_file)
    df.to_csv(current_report_file, mode='a', index=False, header=not file_exists)

print(f"System Active. Logging to: {current_report_file}")

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    
    # Process Detections
    active_detection = None
    if len(results[0].boxes) > 0:
        # Get the detection with the highest confidence
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = CLASS_NAMES.get(cls_id, "Unknown")
        
        # Quality Metrics
        weight = estimate_weight(x2-x1, y2-y1, label)
        freshness, score = check_freshness(frame, x1, y1, x2, y2)
        
        active_detection = {
            "Timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "Type": label,
            "Weight": weight,
            "Freshness": freshness,
            "BBox": (x1, y1, x2, y2)
        }
        history.append(label)

    # UI Overlay
    if active_detection:
        d = active_detection
        x1, y1, x2, y2 = d["BBox"]
        color = (0, 255, 0) if d["Freshness"] == "Fresh" else (0, 0, 255)
        
        # Draw Box and Label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Semi-transparent Info Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"TYPE: {d['Type']}", (10, 30), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, f"WT: {d['Weight']}g", (10, 65), 1, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, f"QUALITY: {d['Freshness']}", (10, 100), 1, 1.5, color, 2)

    # Save data on specific key press (e.g., 's' for Save)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and active_detection:
        log_to_csv(active_detection)
        snapshot_path = os.path.join(SNAPSHOT_DIR, f"onion_{datetime.datetime.now().strftime('%H%M%S')}.jpg")
        cv2.imwrite(snapshot_path, frame)
        print(f"Saved: {active_detection['Type']}")

    cv2.imshow("Onion AI Inspector Pro", frame)
    
    if key == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()