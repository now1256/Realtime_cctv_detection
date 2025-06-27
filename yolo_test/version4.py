import cv2
import threading
import time
from ultralytics import YOLO
rtsp_url = "rtsp://192.168.0.100:554/stream1"
# YOLOv8 모델 로딩 (가벼운 모델 추천)
model = YOLO("yolov8n.pt")

# RTSP 스트림 열기
cap = cv2.VideoCapture(rtsp_url)

frame = None
lock = threading.Lock()

# 🧠 감지 스레드
def detection_loop():
    global frame
    while True:
        time.sleep(1.0)  # 감지 주기 (1초 간격)
        with lock:
            if frame is None:
                continue
            f = frame.copy()

        # YOLOv8 감지 실행
        results = model.predict(source=f, conf=0.5, imgsz=640)

        # 감지 결과 로그 출력
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])
            print(f"🟢 감지됨: {label} (신뢰도: {confidence:.2f})")

# 🎥 영상 출력 메인 루프
def video_loop():
    global frame
    while True:
        ret, new_frame = cap.read()
        if not ret:
            print("❌ RTSP 연결 실패 또는 프레임 수신 실패")
            break

        with lock:
            frame = new_frame.copy()

        cv2.imshow("RTSP Live (YOLO 감지 별도)", new_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔧 감지 스레드 시작
threading.Thread(target=detection_loop, daemon=True).start()

# 🔧 영상 출력 시작
video_loop()
