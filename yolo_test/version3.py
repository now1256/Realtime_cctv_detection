import cv2
import threading
import time
rtsp_url = "rtsp://192.168.0.100:554/stream1"

# MobileNet-SSD 모델 로딩
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/mobilenet_iter_73000.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# RTSP 스트림 열기
cap = cv2.VideoCapture(rtsp_url)

frame = None
lock = threading.Lock()

# 🧠 감지만 수행하는 스레드
def detection_loop():
    global frame
    while True:
        time.sleep(1.0)  # 감지 주기: 1초에 1번
        with lock:
            if frame is None:
                continue
            f = frame.copy()

        # 감지 수행
        blob = cv2.dnn.blobFromImage(f, scalefactor=0.007843, size=(300, 300), mean=127.5)
        net.setInput(blob)
        detections = net.forward()

        # 로그 출력
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                print(f"🟢 감지됨: {label} (신뢰도: {confidence:.2f})")

# 🎥 영상만 띄우는 메인 루프 (RTSP 그대로 표시)
def video_loop():
    global frame
    while True:
        ret, new_frame = cap.read()
        if not ret:
            print("❌ RTSP 연결 끊김 또는 프레임 수신 실패")
            break
        with lock:
            frame = new_frame.copy()

        # 영상만 표시 (감지 결과 표시 X)
        cv2.imshow("RTSP 영상 출력 전용", new_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🔧 감지 스레드 시작
threading.Thread(target=detection_loop, daemon=True).start()

# 🔧 RTSP 영상 출력 시작
video_loop()
