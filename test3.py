import cv2
import numpy as np

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

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_VALUE:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return image

# Video file path
filename = 'images(DL)/1.mp4'

# Open video file
cap = cv2.VideoCapture(filename)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform SSD object detection
    frame = ssdNet(frame)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit if ' ' is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()