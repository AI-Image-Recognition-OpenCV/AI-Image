# AI OpenCV 영상 처리 🧑👩
이 프로젝트는 웹캠을 사용하여 실시간으로 얼굴을 감지하고, 감지된 얼굴 간의 거리를 측정하여 일정 거리 이내로 접근했을 때 경고음을 재생하는 프로그램입니다.

## 기능
+ OpenCV를 사용한 실시간 얼굴 감지.
+ 감지된 얼굴 간의 거리 측정.
+ 얼굴 간 거리가 일정 임계값 이하일 때 경고음 재생.
## 요구 사항
+ Python 3.x
+ OpenCV
+ NumPy
+ gTTS
+ pygame
+ Yolov3
    * yolov3.weights : YOLOv3 모델의 학습된 가중치를 포함.
    * yolov3.cfg : YOLOv3 모델의 아키텍처를 정의.
```
import cv2
import numpy as np
import pyttsx3

#YOLO 객체 감지 함수
def yoloNet(image, net, output_layers, CONF_VALUE=0.5):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    persons_detected = False

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_VALUE and class_id == 0:  # COCO 데이터셋에서 사람의 class_id는 0입니다
                persons_detected = True
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_VALUE, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Person: {confidences[i]:.2f}"
            color = (0, 255, 0)  # 사람의 경우 초록색 사각형
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if persons_detected:
        texttospeech("노란선 안으로 들어가세요.")

    return image

#텍스트를 음성으로 변환하는 함수
def texttospeech(text):
    # pyttsx3 엔진 초기화
    engine = pyttsx3.init()

    # 속성 조정 (속도, 볼륨, 목소리)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume + 0.25)

    # 한국어 목소리 설정 (시스템에 따라 다를 수 있음)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:  # 한국어 목소리 찾기
            engine.setProperty('voice', voice.id)
            break

    # 텍스트를 음성으로 변환
    engine.say(text)
    engine.runAndWait()

#YOLO 모델 로드 및 파라미터 초기화
yolo_weights = 'yolov3.weights'
yolo_cfg = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()

#출력 레이어 이름 가져오기
output_layers = [layer_names[i - 1] for i in output_layers_indexes]

#비디오 파일 경로 (다른 비디오 파일이어야 함)
filename1 = 'images(DL)/2.mp4'
filename2 = 'images(DL)/2.mp4'

#두 개의 비디오 캡처 열기
cap1 = cv2.VideoCapture(filename1)
cap2 = cv2.VideoCapture(filename2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video.")
    exit()

#프레임 스킵 값
frame_skip = 2
frame_count = 0
last_combined_frame = None

#프레임 루프
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    frame_count += 1
    if frame_count % frame_skip == 0:
        # 처리 속도를 높이기 위해 프레임 크기 조정
        small_frame1 = cv2.resize(frame1, (640, 480))
        small_frame2 = cv2.resize(frame2, (640, 480))

        # 두 번째 프레임에 YOLO 객체 감지 수행
        detected_frame = yoloNet(small_frame2, net, output_layers)

        # 원본과 감지된 프레임을 수직으로 결합
        combined_frame = np.vstack((small_frame1, detected_frame))

        # 결합된 프레임 표시
        cv2.imshow('Original (top) and Object Detection (bottom)', combined_frame)
        last_combined_frame = combined_frame  # 마지막으로 처리된 프레임 저장

    # 스페이스바가 눌리면 종료
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

#비디오가 끝나면 마지막 프레임 계속 표시
if last_combined_frame is not None:
    while True:
        cv2.imshow('Original (top) and Object Detection (bottom)', last_combined_frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

#리소스 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
```
## 설명
이 프로젝트는 YOLO 모델을 사용하여 비디오에서 실시간으로 사람을 감지하고, 감지된 사람이 일정 거리 이내로 접근하면 경고음을 재생하는 프로그램입니다.
