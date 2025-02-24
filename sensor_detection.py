from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('runs/detect/train4/weights/best.pt')

# Initialize webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Display the output
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
