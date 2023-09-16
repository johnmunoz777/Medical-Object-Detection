import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import cvzone
import math

# Custom CSS for professional styling
st.markdown(
    """
    <style>
    /* Global styles */
    body {
        background-color: #001F3D; /* Dark Blue background for the app */
    }
    h1, h2, h3 {
        color: #2ECC71 !important; /* Green title text */
    }

    /* Style for settings sidebar */
    .sidebar .stTitle {
        color: #2ECC71 !important; /* Green title text */
    }
    .sidebar .stSlider, .sidebar .stButton {
        background-color: #003366 !important; /* Dark Blue background for slider and button */
        color: white !important; /* White color for text in slider and button */
        font-weight: bold;
        border-radius: 10px;
    }
    .sidebar .stCheckbox, .sidebar .stCheckbox label {
        color: #2ECC71 !important; /* Green color for checkbox and label */
    }
    .sidebar .stCheckbox input:checked + label::before {
        border: 2px solid #2ECC71 !important; /* Green border for checked checkbox */
        background-color: #2ECC71 !important; /* Green background for checked checkbox */
    }
    .sidebar .stButton:hover {
        background-color: #2ECC71; /* Green background for button on hover */
    }

    /* Style for main content */
    .main .stButton {
        background-color: #003366 !important; /* Dark Blue background for button */
        color: white !important; /* White color for button text */
        font-weight: bold;
        border-radius: 10px;
    }
    .main .stAlert {
        background-color: #003366 !important; /* Green background for alert */
        color: white !important; /* White text color for alert */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the title and background color for the Streamlit app
st.markdown("<h1 class='stTitle'>YOLOv8 Medical Video Object Detection</h1>", unsafe_allow_html=True)

# Upload a video
uploaded_file = st.file_uploader("Browse File", type=["mp4"], key="video_upload")

# Create a sidebar for user settings
st.sidebar.markdown("<h1 class='stTitle'>Settings</h1>", unsafe_allow_html=True)

# Style Confidence Threshold and Show Bounding Boxes labels
st.sidebar.subheader("Confidence Threshold")
confidence_threshold = st.sidebar.slider("", 0.0, 1.0, 0.97)
st.sidebar.subheader("Show Bounding Boxes")
show_bounding_boxes = st.sidebar.checkbox("", value=True)

# Replay button
if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()
    cap = cv2.VideoCapture(temp_file.name)
    model = YOLO("best.pt")
    classNames = ['gloves', 'scalpel', 'tube', 'needle', 'gauze', 'tape', 'blanket', 'stretcher', 'lights',
                  'monitor', 'mask', 'iv', 'scrubs', 'gasses', 'instrument']
    video_placeholder = st.empty()

    while True:
        success, img = cap.read()
        if not success:
            cap.release()
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                if conf > confidence_threshold:
                    if show_bounding_boxes:
                        # Increase the size of the bounding box and text
                        x1, y1, x2, y2 = x1 - 10, y1 - 10, x2 + 10, y2 + 10
                        cvzone.cornerRect(img, (x1, y1, w + 20, h + 20), l=10, t=2)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1 + 5), max(10, y1 + 30)), scale=2,
                                       thickness=2, offset=4)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video_placeholder.image(img, use_column_width=True)

    if st.button("Replay Video"):
        cap = cv2.VideoCapture(temp_file.name)
        while True:
            success, img = cap.read()
            if not success:
                cap.release()
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_placeholder.image(img, use_column_width=True)
    os.remove(temp_file.name)
else:
    st.warning("Please upload a video for Medical Object Detection.")

#"https://media.giphy.com/media/YnBntKOgnUSBkV7bQH/giphy.gif"
# Add a GIF image at the end
st.sidebar.image("https://media2.giphy.com/media/0lGd2OXXHe4tFhb7Wh/giphy.gif", use_column_width=True)
st.sidebar.markdown("<p style='color: #2ECC71; font-weight: bold;'>Made by John Munoz </p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color: #2ECC71; font-weight: bold;'>MSDS Regis Graduate Program </p>", unsafe_allow_html=True)
