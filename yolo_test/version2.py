import cv2
rtsp_url = "rtsp://192.168.0.100:554/stream1"
# MobileNet SSD 모델 로드
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/mobilenet_iter_73000.caffemodel"
)
# 클래스 목록 (COCO 기반 일부)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300), mean=127.5)
    net.setInput(blob)
    detections = net.forward()

    # 탐지 결과 루프
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            print(f"🔔 감지됨: {label} (신뢰도: {confidence:.2f})")
            # 한 번만 출력하고 싶다면 여기서 break

    cv2.imshow("MobileNet SSD Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
