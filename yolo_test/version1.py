import cv2
from ultralytics import YOLO
# 2. RTSP URL 설정 (실제 CCTV 주소로 대체)
rtsp_url = "rtsp://192.168.0.100:554/stream1"

import cv2
import threading
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(rtsp_url)

frame = None
annotated_frame = None
lock = threading.Lock()

def yolo_worker():
    global frame, annotated_frame
    while True:
        time.sleep(1.0)  # 1초 간격으로 YOLO 실행
        with lock:
            if frame is None:
                continue
            results = model.predict(source=frame, conf=0.5, imgsz=640)
            annotated_frame = results[0].plot()

# YOLO 백그라운드 스레드 시작
threading.Thread(target=yolo_worker, daemon=True).start()

while True:
    ret, new_frame = cap.read()
    if not ret:
        break

    with lock:
        frame = new_frame.copy()
        display = annotated_frame.copy() if annotated_frame is not None else new_frame.copy()

    cv2.imshow("Smooth CCTV with Async YOLO", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
