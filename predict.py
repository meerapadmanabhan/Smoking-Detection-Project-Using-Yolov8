from ultralytics import YOLO


# Initialize the model
model = YOLO("yolov8m-seg-custom.pt")

model.predict(source="test.mp4", show=True, save=True, show_labels=True,conf=0.5, save_txt=False, save_crop=False, line_width=2)
