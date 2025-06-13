#only log
import cv2
import time
from ultralytics import YOLO

rtsp_url = "rtsp://192.168.0.100:554/stream1"

# YOLOv8 모델 로딩 (Nano 모델 추천)
model = YOLO("yolov8n.pt")

# RTSP 스트림 열기
cap = cv2.VideoCapture(rtsp_url)

# 감지 루프 (영상 출력 없이 로그만)
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 수신 실패")
        break

    # YOLO 감지 수행
    results = model.predict(source=frame, conf=0.5, imgsz=320)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])

        # 로그만 출력
        print(f"🟢 감지됨: {label} (신뢰도: {confidence:.2f})")