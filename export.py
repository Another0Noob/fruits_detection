from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11l.pt")

# Export the model to ONNX format and save it in the root directory
model.export(format="onnx")  # creates 'yolo11n.onnx'

onnx_model = YOLO("yolo11l.onnx")

# Save the model in the root directory
