import cv2
import numpy as np
import pyttsx3

# Function to perform text-to-speech
def texttospeech(text):
    # pyttsx3 engine initialization
    engine = pyttsx3.init()

    # Adjust properties (speed, volume, voice, etc.)
    rate = engine.getProperty('rate')   # Default speed
    engine.setProperty('rate', rate - 50)  # Decrease speed
    volume = engine.getProperty('volume')  # Default volume
    engine.setProperty('volume', volume + 0.25)  # Increase volume

    # Set Korean voice (may vary depending on the system)
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'ko' in voice.id:  # Find Korean voice
            engine.setProperty('voice', voice.id)
            break

    # Convert text to speech
    engine.say(text)
    engine.runAndWait()

# Function for SSD object detection
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

# Variable initialization
capture = cv2.VideoCapture(0)  # Capture from webcam (change as needed)
s_factor = 0.5
confValue = 0.5
ALERT_DISTANCE_CM = 50

# Window setup
window_name = 'AI - Person recognition and distance measurement'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 700, 550)

# Main loop for object detection
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

                # Call text-to-speech function for alert
                if distance_cm < ALERT_DISTANCE_CM:
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
