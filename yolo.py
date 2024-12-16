from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("data/ski.mp4", show=True)
