import cv2
import numpy as np
import pyttsx3
import pygame
import time
import os
from test import texttospeech


# 텍스트를 음성으로 변환하여 재생하는 함수
def text_to_speech(text, lang='ko'):
    # pyttsx3 엔진 초기화
    engine = pyttsx3.init()

    # 속성 설정 (속도, 볼륨, 목소리 등)
    rate = engine.getProperty('rate')  # 기본 속도 가져오기
    engine.setProperty('rate', rate - 50)  # 속도 줄이기
    volume = engine.getProperty('volume')  # 기본 볼륨 가져오기
    engine.setProperty('volume', volume + 0.25)  # 볼륨 높이기

    # 한국어 음성 설정 (시스템에 따라 다를 수 있음)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:  # 한국어 음성 찾기
            engine.setProperty('voice', voice.id)
            break

    # 텍스트를 음성으로 변환
    engine.say(text)
    engine.runAndWait()

# 객체 감지 함수
def ssdNet(image, confVal):
    CONF_VALUE = confVal
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor", "book", "phone"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    persons = []

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if class_id == 15 and confidence > CONF_VALUE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[class_id], 2)
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)
            persons.append((centerX, centerY))

    return image, persons

# 변수 설정
capture = cv2.VideoCapture(0)

s_factor = 0.5
confValue = 0.5
ALERT_DISTANCE_CM = 50

# 창 이름 설정 및 크기 조정
window_name = 'AI - Person recognition and distance measurement'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 700, 550)

# AI 객체 감지 실행
frameCount = 0
while True:
    ret, frame = capture.read()
    if not ret:
        break

    frameCount += 1
    if frameCount % 5 == 0:
        frame = cv2.resize(frame, None, fx=s_factor, fy=s_factor, interpolation=cv2.INTER_AREA)
        detected_frame, persons = ssdNet(frame, confValue)

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                centerX1, centerY1 = persons[i]
                centerX2, centerY2 = persons[j]

                distance_px = np.sqrt((centerX1 - centerX2) ** 2 + (centerY1 - centerY2) ** 2)

                pixel_to_cm_ratio = 0.026458
                distance_cm = distance_px * pixel_to_cm_ratio

                distance_str = "{:.2f} cm".format(distance_cm)
                cv2.putText(detected_frame, distance_str, (centerX1, centerY1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

                if __name__ == "__main__":
                    text = "노란선 안으로 들어가세요."
                    texttospeech(text)


        cv2.imshow(window_name, detected_frame)


    key = cv2.waitKey(20)
    if key == ord(' '):
        break
    elif key == ord('b'):
        break


capture.release()
cv2.destroyAllWindows()
