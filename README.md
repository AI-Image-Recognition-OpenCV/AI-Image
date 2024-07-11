# AI OpenCV 영상 처리 
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


import cv2
import numpy as np
import pyttsx3

# Function for YOLO object detection
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
            if confidence > CONF_VALUE and class_id == 0:  # Only detect persons (class_id == 0 for COCO dataset)
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
            color = (0, 255, 0)  # Green color for persons
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if persons_detected:
        texttospeech("노란선 안으로 들어가세요.")

    return image

# Function for text to speech
def texttospeech(text):
    # pyttsx3 engine initialization
    engine = pyttsx3.init()

    # Adjusting properties (rate, volume, voice)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume + 0.25)

    # Setting Korean voice (may vary depending on the system)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:  # Finding Korean voice
            engine.setProperty('voice', voice.id)
            break

    # Converting text to speech
    engine.say(text)
    engine.runAndWait()

# Load YOLO model and initialize parameters
yolo_weights = 'yolov3.weights'
yolo_cfg = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()

# Get the names of the output layers
output_layers = [layer_names[i - 1] for i in output_layers_indexes]

# Video file paths (should be different videos)
filename1 = 'images(DL)/2.mp4'
filename2 = 'images(DL)/2.mp4'

# Open two video captures
cap1 = cv2.VideoCapture(filename1)
cap2 = cv2.VideoCapture(filename2)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame skip value
frame_skip = 2
frame_count = 0
last_combined_frame = None

# Loop through frames
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    frame_count += 1
    if frame_count % frame_skip == 0:
        # Resize the frames to improve processing speed
        small_frame1 = cv2.resize(frame1, (640, 480))
        small_frame2 = cv2.resize(frame2, (640, 480))

        # Perform YOLO object detection on the second frame
        detected_frame = yoloNet(small_frame2, net, output_layers)

        # Combine the original and detected frames vertically
        combined_frame = np.vstack((small_frame1, detected_frame))

        # Display the combined frame
        cv2.imshow('Original (top) and Object Detection (bottom)', combined_frame)
        last_combined_frame = combined_frame  # Save the last processed frame

    # Exit if spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# If the video ends, keep showing the last frame
if last_combined_frame is not None:
    while True:
        cv2.imshow('Original (top) and Object Detection (bottom)', last_combined_frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
