import onnxruntime as ort
import numpy as np
import cv2


# ONNX 모델 불러오기
session = ort.InferenceSession("yolov8n.onnx", providers=["DmlExecutionProvider"])
input_name = session.get_inputs()[0].name

# 이미지 전처리
img = cv2.imread("C:/Users/nabis/Desktop/images.jpg")
img = cv2.resize(img, (224, 224))
img = img[:, :, ::-1]  # BGR → RGB
img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

# 추론
outputs = session.run(None, {input_name: img})
pred = np.argmax(outputs[0])
print("Predicted class ID:", pred)