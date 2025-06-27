import cv2
import numpy as np
import onnxruntime as ort
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

# ONNX Runtime 세션 (AMD GPU를 사용하는 경우 DmlExecutionProvider)
session = ort.InferenceSession("yolov8n.onnx", providers=["DmlExecutionProvider"])
input_name = session.get_inputs()[0].name

# 비디오 경로
video_path = "C:/Users/nabis/Downloads/MVUydlRlWmtsYndfNzIwcA_out.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 비디오 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 입력 전처리
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = img_resized[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_rgb, axis=0)

    # ONNX 추론
    start = time.time()
    output = session.run(None, {input_name: img_input})[0]  # (1, 300, 6)
    elapsed = time.time() - start

    for det in output[0]:
        x1, y1, x2, y2, conf, class_id = det
        if conf < 0.5:
            continue

        label = CLASS_NAMES[int(class_id)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Inference time: {elapsed:.3f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 ONNX Detection", frame)

    # ESC 눌러 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
