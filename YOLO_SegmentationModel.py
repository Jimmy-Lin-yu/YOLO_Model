from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # 或 yolo11n.yaml
print(model.model)

