import cv2
import numpy as np
import pyttsx3

# 인화성 물질 리스트 (예: 흡연, 라이터, 폭탄 등)
TARGET_CLASSES = ["cigarette", "lighter", "gasoline", "explosive"]

# SSD 객체 감지 함수
def ssdNet(image, net, ln, LABELS):
    CONF_VALUE = 0.5
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_objects = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            label = LABELS[idx]
            if label in TARGET_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, f"{label}: {confidence:.2f}", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                detected_objects.append(label)

    return image, detected_objects

# 텍스트 음성 변환 함수
def texttospeech(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume + 0.25)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

# 사전 훈련된 모델 및 클래스 로드
prototxt_path = 'MobileNetSSD_deploy.prototxt.txt'
model_path = 'MobileNetSSD_deploy.caffemodel'
LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
          "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
          "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
          "sofa", "train", "tvmonitor", "cigarette", "lighter", "gasoline", "explosive"]

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 비디오 파일 경로
filename = 'images(DL)/2.mp4'
cap1 = cv2.VideoCapture(filename)
cap2 = cv2.VideoCapture(filename)

if not (cap1.isOpened() and cap2.isOpened()):
    print("Error: Could not open video.")
    exit()

frame_skip = 2
frame_count = 0
last_frame = None
s_factor = 0.5 # 화면 크기 비율 (조절 가능)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not (ret1 and ret2):
        break

    frame_count += 1
    frame1 = cv2.resize(frame1, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
    frame2 = cv2.resize(frame2, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)

    if frame_count % frame_skip == 0:
        detected_frame, detected_objects = ssdNet(frame2, net, None, LABELS)

        if detected_objects:
            texttospeech("경고! 인화성 물질이 감지되었습니다.")

        combined_frame = np.vstack((frame1, detected_frame))
        last_frame = combined_frame

        cv2.imshow('Original and Object Detection', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# 영상이 끝난 후 마지막 프레임을 계속 표시
if last_frame is not None:
    while True:
        cv2.imshow('Original and Object Detection', last_frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()