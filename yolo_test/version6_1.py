import cv2
import time
import multiprocessing as mp
from ultralytics import YOLO

# RTSP 스트림 주소
rtsp_url = "rtsp://192.168.0.100:554/stream1"

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
            print(f"🟢 감지됨: {label} (신뢰도: {confidence:.2f})", flush=True)

# 메인 프로세스: 프레임 캡처 + 영상 출력
def main():
    mp.set_start_method('spawn', force=True)  # Windows 호환성

    input_queue = mp.Queue(maxsize=5)
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
                print("❌ 프레임 수신 실패")
                break

            # 감지 프로세스로 프레임 전송
            if not input_queue.full():
                input_queue.put(frame)

            # 원본 영상 출력
            cv2.imshow("📷 RTSP Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
                print("🛑 q 키 입력: 종료 요청됨", flush=True)
                break

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
            print("⚠️ 프로세스 강제 종료 시도 중...", flush=True)
            worker.terminate()

        worker.join()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 정상 종료됨", flush=True)

if __name__ == "__main__":
    main()
