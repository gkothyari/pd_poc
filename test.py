import kagglehub

# Download latest version
path = kagglehub.dataset_download("anggadwisunarto/potholes-detection-yolov8")

print("Path to dataset files:", path)