from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/onion_model/weights/best.pt")

print("Class Names:", model.names)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25)

    annotated = results[0].plot()
    cv2.imshow("Test Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()