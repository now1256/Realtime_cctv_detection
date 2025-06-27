import cv2
import time
import multiprocessing as mp
from ultralytics import YOLO

# RTSP 스트림 주소
rtsp_url = "rtsp://192.168.0.12:554/stream1"

# 감지 프로세스 (YOLO 로드 + 감지만 수행)
def detect_worker(input_queue):
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano 모델 (CPU 사용)

    while True:
        frame = input_queue.get()
        if frame is None:  # 종료 신호
            break
        results = model.predict(source=frame, conf=0.5, imgsz=320, verbose=False)
       

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])
            print(f"🟢 감지됨: {label} (신뢰도: {confidence:.2f})")

        


# 메인 프로세스: 프레임 캡처
def main():
    mp.set_start_method('spawn', force=True)  # Windows 호환성

    input_queue = mp.Queue(maxsize=1)  # 항상 최신 1개만 유지
    worker = mp.Process(target=detect_worker, args=(input_queue,))
    worker.start()

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ RTSP 스트림 열기 실패")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 프레임 수신 실패")
                time.sleep(0.1)
                continue

            # 이전 프레임 제거 (항상 최신 프레임만 유지)
            if not input_queue.empty():
                try:
                    input_queue.get_nowait()
                except:
                    pass

            # 최신 프레임 삽입
            try:
                input_queue.put_nowait(frame)
            except:
                pass  # 큐가 가득 찼을 때는 무시 (사실상 불가능)

    except KeyboardInterrupt:
        print("🛑 Ctrl+C 감지: 종료 요청됨")

    finally:
        print("⛔ 종료 중...")

        try:
            input_queue.put_nowait(None)
        except:
            pass

        time.sleep(1)

        if worker.is_alive():
            print("⚠️ 프로세스 강제 종료 시도 중...")
            worker.terminate()

        worker.join()
        cap.release()
        print("✅ 정상 종료됨")

if __name__ == "__main__":
    main()
