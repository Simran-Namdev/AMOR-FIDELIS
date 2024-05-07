# Amor Fidelis: Nurturing Connection Through AI for the Visually Impaired

## Overview:
Amor Fidelis is an innovative project aimed at addressing the challenges faced by visually impaired individuals, particularly children, in navigating social interactions and accessing information independently. Leveraging the power of Artificial Intelligence (AI) and expert systems, this project provides real-time auditory feedback derived from visual analysis to enhance the user's awareness and empower them to navigate complex environments with greater autonomy. The project focuses on seamless integration with existing senses, prioritizing user experience.

## Features:
- Object Detection: Utilizes a pre-trained YOLOv3 model to detect objects in images or via webcam in real-time.
- Text Recognition: Recognizes and reads text from images.
- Text-to-Speech Conversion: Converts recognized text into audible speech for users.
- Knowledge Base: Utilizes an inference engine expert system for object description, providing users with factual information about detected objects.

## Prerequisites:
- YOLO Weights: Download the YOLOv3 pre-trained weights and save them in the project directory.
- YOLO Model: Ensure you have the YOLOv3 model configuration file (`yolov3.cfg`) and class names file (`coco.names`) in the project directory.

## Installation:
1. Clone the repository: `https://github.com/Simran-Namdev/AMOR-FIDELIS`
2. Install required dependencies: `pip install -r requirements.txt`

## Usage:
1. Ensure you have Python and necessary dependencies installed.
2. Navigate to the project directory.
3. Run the Streamlit UI script: `streamlit run streamlit_ui.py`
4. Choose the input option: either upload an image or use the webcam.
5. If uploading an image, select an image file. If using the webcam, grant access to the webcam.
6. Detected objects will be highlighted, and their descriptions will be provided.
7. Enjoy exploring and learning about various objects present in the image!

## Project Structure:
- `yolomodel.py`: Contains the YOLO model implementation for object detection.
- `facts_knowledge_base.py`: Implements the knowledge base for object descriptions.
- `streamlit_ui.py`: Streamlit user interface script for interacting with the project.
- `requirements.txt`: List of required Python dependencies.
- `README.md`: Project overview, installation instructions, and usage guide.
- `images/`: Directory containing sample images for testing.

## License:
 MIT License
