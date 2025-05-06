from ultralytics import YOLO

# Load your YOLO model
model = YOLO("yolo11l.pt")  # replace with your model path if different

# Export the model to ONNX format
model.export(format="onnx")