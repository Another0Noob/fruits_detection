import cv2
from ultralytics import YOLO

# Load the YOLO model
model_path = "yolo11l.pt"  # Update this path if needed
model = YOLO(model_path)

def detect_and_draw(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Perform detection
    results = model(image)

    # Iterate through detections
    for result in results[0].boxes:
        # Extract bounding box, class, and confidence
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        conf = result.conf[0]  # Confidence score
        cls = result.cls[0]  # Class index
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Class name and confidence

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "img.jpg"  # Replace with the path to your image
detect_and_draw(image_path)