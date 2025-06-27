import cv2
import onnxruntime
import numpy as np
import time

# COCO 클래스 (80개)
CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# ONNX 모델 로딩 (AMD GPU용)
session = onnxruntime.InferenceSession("yolov8n.onnx", providers=["DmlExecutionProvider"])
input_name = session.get_inputs()[0].name

# RTSP 스트림 열기
rtsp_url = "rtsp://192.168.0.161:554/stream1"  # ← 실제 RTSP 주소로 변경
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("RTSP 스트림을 열 수 없습니다.")
    exit()

print("✅ RTSP 연결 성공. 실시간 감지 시작...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임을 가져오지 못했습니다. 다시 시도 중...")
        time.sleep(0.5)
        continue

    # YOLO 입력용 전처리
    img = cv2.resize(frame, (640, 640))
    img_rgb = img[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_rgb, axis=0)

    # ONNX 추론
    outputs = session.run(None, {input_name: img_input})[0]  # (1, 300, 6)

    for det in outputs[0]:
        x1, y1, x2, y2, conf, class_id = det
        if conf < 0.5:
            continue

        label = CLASS_NAMES[int(class_id)]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 감지됨: {label} ({conf:.2f}) "
              f"위치: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    # 필요 시 ESC 눌러 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
