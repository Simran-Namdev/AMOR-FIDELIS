# yolomodel.py
import cv2 as cv
import numpy as np

class YoloModel:
    def __init__(self, weights_path, config_path, classes_path):
        self.net = cv.dnn.readNet(weights_path, config_path)
        self.classes = self.load_classes(classes_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def detect_objects(self, frame):
        height, width, channels = frame.shape
        blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

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

        detected_objects = []

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, label, (x, y + 30), cv.FONT_HERSHEY_PLAIN, 3, color, 3)

                detected_objects.append(label)

        return frame, detected_objects