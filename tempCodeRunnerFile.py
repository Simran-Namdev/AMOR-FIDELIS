import cv2 as cv
import numpy as np
from wandb import Classes
import os

# load yolo to take objects in image as rules
net = cv.dnn.readNet("yolov3.weights",
                     "yolov3.cfg")
clasees = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv.imread("room_ser.jpeg")
img = cv.resize(img, None, fx=0.4, fy=0.4)
height, width, channel = img.shape

blob = cv.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layer)
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y + 30), font, 3, color, 3)

cv.imshow("IMG", img)
cv.waitKey(0)
cv.destroyAllWindows()

detected_objects = []

for i in range(len(boxes)):
    if i in indexes:
        label = str(classes[class_ids[i]])
        detected_objects.append(label)
        
#Linking Objects(Rules) to facts

facts_directory = "facts"

object_descriptions = {}
for filename in os.listdir(facts_directory):
    if filename.endswith(".txt"):
        with open(os.path.join(facts_directory, filename), 'r') as file:
            object_name = os.path.splitext(filename)[0]
            description = file.read().strip()
            object_descriptions[object_name] = description
matched_objects = {}
for obj in detected_objects:
    if obj in object_descriptions:
        matched_objects[obj] = object_descriptions[obj]

for obj, description in matched_objects.items():
    print("Object:", obj)
    print("Description:", description)
    print()
