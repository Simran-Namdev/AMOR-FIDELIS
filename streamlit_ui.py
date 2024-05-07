import streamlit as st
import cv2 as cv
import numpy as np
import pyttsx3
from yolomodel import YoloModel
from facts_knowledge_base import FactsKnowledgeBase

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load YOLO to detect objects in the image
yolo_model = YoloModel("yolov3.weights", "yolov3.cfg", "coco.names")
facts_knowledge_base = FactsKnowledgeBase("facts")

# Function to perform object detection on a single frame
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_model.net.setInput(blob)
    outs = yolo_model.net.forward(yolo_model.output_layers)

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
            label = str(yolo_model.classes[class_ids[i]])
            color = yolo_model.colors[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, label, (x, y + 30), cv.FONT_HERSHEY_PLAIN, 3, color, 3)

            detected_objects.append(label)  # Append only the label

    return frame, detected_objects

# Function to handle text-to-speech
def speak_description(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Object Detection")
option = st.radio("Choose Input Option:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)

        st.image(img, channels="BGR")

        img_with_detections, detected_objects = detect_objects(img)

        st.subheader("Detected Objects:")
        detected_object_labels = detected_objects  # No need to extract labels separately
        ini = "I see: " + ", ".join(detected_object_labels) # Speak all detected objects
        for label in detected_object_labels:
            st.write(f"- {label}")

        st.write("---")
        speech = ini + ". I would also like to tell you more about "
        for label in detected_object_labels:
            description = facts_knowledge_base.get_description(label)
            if description:
                speech += f"{label}: {description}. "
                st.write(f"**Description of {label}:** {description}")

        st.write("---")
        stop_option = st.button("Stop and Select Another Image", key="stop_image_button")
        if stop_option:
            st.experimental_rerun()
            
        speak_description(speech)
else:
    st.write("Waiting for webcam access...")

    # Initialize webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        st.write("Error: Couldn't access webcam.")
        st.stop()

    stop_camera = False
    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        frame_with_detections, detected_objects = detect_objects(frame)

        st.image(frame_with_detections, channels="BGR", caption="Object Detection Result")

        st.subheader("Detected Objects:")
        detected_object_labels = detected_objects  # No need to extract labels separately
        ini = "I see: " + ", ".join(detected_object_labels) # Speak all detected objects
        for label in detected_object_labels:
            st.write(f"- {label}")

        st.write("---")
        speech = ini + ". I would also like to tell you more about "
        for label in detected_object_labels:
            description = facts_knowledge_base.get_description(label)
            if description:
                speech += f"{label}: {description}. "
                st.write(f"**Description of {label}:** {description}")

        st.write("---")
        stop_camera = st.button("Stop Camera", key="stop_camera_button")
        if stop_camera:
            break
        speak_description(speech)

    cap.release()
    cv.destroyAllWindows()
