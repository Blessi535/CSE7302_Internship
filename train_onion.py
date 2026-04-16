from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on your dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="onion_model"
)

print("Training Completed Successfully!")