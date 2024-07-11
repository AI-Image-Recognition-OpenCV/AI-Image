import cv2
import numpy as np
import pyttsx3
import threading

# Function for SSD object detection
def ssdNet(image):
    CONF_VALUE = 0.5
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load pre-trained SSD model
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    persons_detected = False

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":  # Only detect persons
                persons_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    if persons_detected:
        texttospeech("노란선 안으로 들어가세요.")

    return image

# Function for text to speech
def texttospeech(text):
    # pyttsx3 engine initialization
    engine = pyttsx3.init()

    # Adjusting properties (rate, volume, voice)
    rate = engine.getProperty('rate')   # Getting default rate
    engine.setProperty('rate', rate - 50)  # Decreasing rate
    volume = engine.getProperty('volume')  # Getting default volume
    engine.setProperty('volume', volume + 0.25)  # Increasing volume

    # Setting Korean voice (may vary depending on the system)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:  # Finding Korean voice
            engine.setProperty('voice', voice.id)
            break

    # Converting text to speech
    engine.say(text)
    engine.runAndWait()

# Video file path
filename = 'images(DL)/2.mp4'

# Open video file
cap = cv2.VideoCapture(filename)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame skip value
frame_skip = 2
frame_count = 0
last_frame = None

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip == 0:
        # Resize the frame to improve processing speed
        small_frame = cv2.resize(frame, (640, 480))

        # Perform SSD object detection
        detected_frame = ssdNet(small_frame)

        # Display the frame
        cv2.imshow('Object Detection', detected_frame)
        last_frame = detected_frame  # Save the last processed frame

    # Exit if spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# If the video ends, keep showing the last frame
if last_frame is not None:
    while True:
        cv2.imshow('Object Detection', last_frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()