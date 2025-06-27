from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # yolov8s.pt, yolov8m.pt도 가능
model.export(format="onnx", nms=True)  # DirectML 호환용