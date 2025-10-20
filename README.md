# YOLO 11 Fruit Detection Model Project

This project utilizes the YOLO 11 deep learning model to detect and classify various fruits in images. YOLO 11 is the latest evolution in the "You Only Look Once" object detection family, known for its speed and accuracy. The model has been trained specifically to recognize supermarket fruits and vegetables, making it ideal for retail automation, inventory management, or educational purposes.

For a full demonstration and example code, see our Kaggle notebook: [Fruits & Vegetable Detection with YOLO 11](https://www.kaggle.com/code/another0noob/fruits-vegetable-detection-yolo11).

## Requirements
- Python 3.8 or higher is recommended.
- Python 3.13 has some issues with ONNX.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
   
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The model is trained on the following public dataset:

EndeXspace. "Supermarket Items (YOLOv7) Dataset." Roboflow Universe, Jan. 2025, https://universe.roboflow.com/endexspace/supermarket-items-yolov7. Accessed 2 Aug. 2025.

---

For more information and hands-on examples, visit [Kaggle notebook](https://www.kaggle.com/code/another0noob/fruits-vegetable-detection-yolo11).
