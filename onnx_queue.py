import cv2
import time
import numpy as np
import multiprocessing as mp
import onnxruntime

# RTSP 주소
rtsp_url = "rtsp://192.168.0.161:554/stream1"

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

# 추론 프로세스 함수
def detect_worker(input_queue):
    session = onnxruntime.InferenceSession("yolov8n.onnx", providers=["DmlExecutionProvider"])
    input_name = session.get_inputs()[0].name

    while True:
        frame = input_queue.get()
        if frame is None:
            break

        # 전처리
        img = cv2.resize(frame, (640, 640))
        img_rgb = img[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_rgb, axis=0)

        # 추론
        outputs = session.run(None, {input_name: img_input})[0]  # (1, 300, 6)
        for det in outputs[0]:
            x1, y1, x2, y2, conf, class_id = det
            if conf < 0.5:
                continue

            label = CLASS_NAMES[int(class_id)]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🟢 감지됨: {label} ({conf:.2f}) "
                  f"→ 좌표: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]", flush=True)

# 메인 프로세스 (RTSP 스트림 수신)
def main():
    mp.set_start_method('spawn', force=True)  # Windows 호환

    input_queue = mp.Queue(maxsize=5)
    worker = mp.Process(target=detect_worker, args=(input_queue,))
    worker.start()

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ RTSP 연결 실패")
        return

    print("✅ RTSP 연결 성공. 프레임 수신 시작...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 프레임 수신 실패")
                continue

            if not input_queue.full():
                input_queue.put(frame)

    except KeyboardInterrupt:
        print("🛑 Ctrl+C 감지: 종료 요청됨", flush=True)

    finally:
        print("⛔ 종료 중...", flush=True)
        try:
            input_queue.put_nowait(None)
        except:
            pass

        time.sleep(1)

        if worker.is_alive():
            print("⚠️ 감지 프로세스 강제 종료 중...", flush=True)
            worker.terminate()

        worker.join()
        cap.release()
        print("✅ 전체 종료 완료", flush=True)

if __name__ == "__main__":
    main()
