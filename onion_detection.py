import cv2
from ultralytics import YOLO
from datetime import datetime, timedelta
import os
import numpy as np

# Load model
model = YOLO("runs/detect/onion_model/weights/best.pt")
cap = cv2.VideoCapture(0)

window_name = "Onion Quality Inspection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Stability variables
stable_data = None
last_update_time = None
STABLE_DURATION = 60  # seconds (1 minute)

os.makedirs("onion_reports", exist_ok=True)
os.makedirs("onion_snapshots", exist_ok=True)

def estimate_weight(w, h):
    return round((w * h) * 0.002, 1)

def check_freshness(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "Unknown", (255, 255, 255)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    brightness = np.mean(hsv[:, :, 2])
    saturation = np.mean(hsv[:, :, 1])

    # Improved condition
    if brightness > 120 and saturation > 60:
        return "Fresh", (0, 255, 0)  # Green
    else:
        return "Not Fresh", (0, 0, 255)  # Red

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(source=frame, conf=0.6, verbose=False)

    detected_data = None

    if result[0].boxes and len(result[0].boxes) > 0:
        onion_boxes = [
            b for i, b in enumerate(result[0].boxes)
            if int(result[0].boxes.cls[i]) == 0
        ]

        if onion_boxes:
            b = onion_boxes[0]
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            weight = estimate_weight(w, h)

            size = "Small" if weight < 150 else "Medium" if weight < 300 else "Large"

            freshness, color = check_freshness(frame, x1, y1, x2, y2)

            detected_data = (x1, y1, x2, y2, w, h, weight, size, freshness, color)

            # 🔥 Stability Logic (1 minute lock)
            current_time = datetime.now()

            if stable_data is None:
                stable_data = detected_data
                last_update_time = current_time
            else:
                if (current_time - last_update_time) > timedelta(seconds=STABLE_DURATION):
                    stable_data = detected_data
                    last_update_time = current_time

    # Display stable data
    if stable_data:
        x1, y1, x2, y2, w, h, weight, size, freshness, color = stable_data

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"Width: {w}px", (x1, y1 - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Height: {h}px", (x1, y1 - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Weight: {weight} g", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"{freshness} | {size}", (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    else:
        cv2.putText(frame, "No Onion Detected", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()