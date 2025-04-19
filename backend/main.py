import torch
import cv2
from pathlib import Path
import sys

# Add yolov5 folder to sys.path
FILE = Path(__file__).resolve()
YOLO_DIR = FILE.parents[0] / "yolov5"
sys.path.append(str(YOLO_DIR))

# Load model
model = torch.hub.load('backend/yolov5', 'custom', path='backend/best.pt', source='local')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Draw results
    result_frame = results.render()[0]

    # Show results
    cv2.imshow("Weld/Paint Defect Detection", result_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
