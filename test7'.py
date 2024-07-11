import cv2
import numpy as np
import pyttsx3

# Function for SSD object detection
def ssdNet(image, net, CLASSES, COLORS, CONF_VALUE=0.5):
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

# Load pre-trained SSD model and initialize parameters
prototxt_path = 'MobileNetSSD_deploy.prototxt.txt'
model_path = 'MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Video file path
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
        # Resize the frame to improve processing speed
        small_frame1 = cv2.resize(frame1, (640, 480))
        small_frame2 = cv2.resize(frame2, (640, 480))

        # Perform SSD object detection on the second frame
        detected_frame = ssdNet(small_frame2, net, CLASSES, COLORS)

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
