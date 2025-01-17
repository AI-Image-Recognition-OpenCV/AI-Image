# AI OpenCV 영상 처리 👨👩
이 프로젝트는 웹캠을 사용하여 실시간으로 얼굴을 감지하고, 감지된 얼굴 간의 거리를 측정하여 일정 거리 이내로 접근했을 때 경고음을 재생하는 프로그램입니다.

## 기능⚙️
+ OpenCV를 사용한 실시간 얼굴 감지.
+ 감지된 얼굴 간의 거리 측정.
+ 얼굴 간 거리가 일정 임계값 이하일 때 경고음 재생.
## 요구 사항🦾
+ Python 3.x
+ OpenCV
+ NumPy
+ gTTS
+ pygame
+ Yolov3
    * yolov3.weights : YOLOv3 모델의 학습된 가중치를 포함.
    * yolov3.cfg : YOLOv3 모델의 아키텍처를 정의.
# final(end).py 코드 해석
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
```
이미지를 YOLO 네트워크가 처리할 수 있는 형태로 변환하고, 네트워크를 통해 객체를 감지합니다. 감지된 결과는 outs 변수에 저장됩니다. 이후 단계에서는 이 결과를 해석하여 이미지에서 객체를 표시하거나 다른 처리를 수행할 수 있습니다.
```
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
```
이미지에서 사람을 감지하고, 감지된 사람의 위치를 시각적으로 표시하며, 사람이 감지되었을 때 음성 경고를 출력하는 기능을 구현입니다
```
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
```
 pyttsx3를 초기화하여 한국어 음성으로 텍스트를 변환하고 출력합니다. 목소리 속도와 볼륨을 조정하는 기능이 있습니다. 시스템에 따라 다른 한국어 목소리를 선택할 수 있습니다.
```
#YOLO 모델 로드 및 파라미터 초기화
yolo_weights = 'yolov3.weights'
yolo_cfg = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()
```
위 코드는 YOLOv3 모델을 로드하고 초기화하는 과정을 요약한 것입니다:

+ yolo_weights와 yolo_cfg는 YOLOv3 모델의 가중치 파일과 설정 파일을 지정합니다.
+ cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)을 사용하여 Darknet 형식의 YOLOv3 모델을 로드합니다.
+ net.getLayerNames()를 통해 네트워크의 모든 레이어 이름을 가져오고, net.getUnconnectedOutLayers()로 출력 레이어의 인덱스를 가져옵니다.
```
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
```
위 코드는 두 개의 비디오 파일을 열어 YOLO 객체 감지를 수행하고, 감지된 객체를 원본 비디오와 함께 화면에 표시하는 과정을 요약한 것입니다:

output_layers 변수는 YOLO 네트워크의 출력 레이어 이름 목록을 저장합니다.

+ cap1과 cap2는 각각 filename1과 filename2의 비디오 파일을 엽니다.
+ 비디오 파일이 열리지 않으면 오류 메시지를 출력하고 프로그램을 종료합니다.
+ frame_skip 프레임마다 YOLO 객체 감지를 수행하고, 감지된 프레임을 원본과 수직으로 결합하여 combined_frame에 저장합니다.
+ 사용자가 스페이스바를 누르면 프로그램을 종료하며, 비디오의 끝에 도달하면 마지막으로 처리된 프레임을 계속 표시합니다.

  
이 코드는 실시간 객체 감지를 비디오에 적용하고, 그 결과를 사용자에게 시각적으로 제공하는 데 사용될 수 있습니다.
```
#리소스 해제
cap1.release()
cap2.release()
cv2.destroyAllWindows()
```

비디오 캡처와 화면에 표시된 모든 창을 안전하게 해제하는 과정을 요약한 것입니다:

+ cap1.release(): 첫 번째 비디오 캡처 리소스를 해제합니다.
+ cap2.release(): 두 번째 비디오 캡처 리소스를 해제합니다.
+ cv2.destroyAllWindows(): 현재 열린 모든 OpenCV 창을 닫습니다.
이 과정은 프로그램 실행을 마치고 난 후, 모든 리소스를 정리하여 메모리 누수를 방지합니다.

## 설명🖐🏻
이 프로젝트는 YOLO 모델을 사용하여 비디오에서 실시간으로 사람을 감지하고, 감지된 사람이 일정 거리 이내로 접근하면 경고음을 재생하는 프로그램입니다.
