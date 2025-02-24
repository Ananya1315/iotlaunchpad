from ultralytics import YOLO

# Load the YOLO model
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

# Perform a dry run to check the dataset without actual training
print("Checking dataset...")
model.train(data='dataset/dataset.yaml', epochs=0)
print("Dataset check completed!")
