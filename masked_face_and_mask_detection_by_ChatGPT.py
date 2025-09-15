import streamlit as st

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="Masked-Face-Recognition", page_icon="😷", layout="wide")

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration, WebRtcMode
from streamlit_option_menu import option_menu
import av
import queue
import threading
from typing import List, NamedTuple, Union
import time

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    model1 = tf.keras.models.load_model('masked_face_detector.h5')
    model2 = tf.keras.models.load_model('mask_detector.h5')
    return model1, model2

model1, model2 = load_models()

# -------------------------------
# Face detection function
# -------------------------------
def dnn_extract_face(img):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    labels = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            face = img[startY:endY, startX:endX]
            if face.size == 0:
                continue
            # Mask detection
            face_resized = cv2.resize(face, (350, 350))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = np.expand_dims(np.array(Image.fromarray(face_rgb)), axis=0)
            mask_pred = model2.predict(face_array, verbose=0)[0]
            mask, withoutMask = mask_pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 235, 0)
            label_text = f"{label}: {max(mask, withoutMask) * 100:.1f}%"
            cv2.putText(img, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            faces.append(face)
            labels.append(label_text)
    return faces, labels, img

# -------------------------------
# Faces database
# -------------------------------
faces_db = ['Abdur Samad', 'Ahsan Ahmed', 'Asef', 'Ashik', 'Azizul Hakim', 'DDS', 'Mahmud',
            'Mayaz', 'Meheraj', 'Nayeem Khan', 'Nayem', 'Rezwanul Huq', 'Risul Islam Fahim',
            'Saif', 'Saki', 'Samir', 'Shahtab', 'Shamim H Ripon', 'Shimul Rahman Fahad',
            'Shourov', 'Shuvo', 'Sizan']

# -------------------------------
# RTC config
# -------------------------------
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

choice = option_menu("Masked Face Recognition Application",
                     ["Upload Image", "Take a Snapshot", "Real Time Detection", "About"],
                     icons=['file-earmark-arrow-up', 'camera', 'camera-video', 'house'],
                     menu_icon="emoji-smile", default_index=0, orientation="horizontal",
                     styles={
                         "container": {"background-color": "#002522"},
                         "icon": {"color": "orange", "font-size": "25px"},
                         "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                      "--hover-color": "#ff2d00"},
                         "nav-link-selected": {"background-color": "#02ab21"},
                     })

# -------------------------------
# Upload Image
# -------------------------------
if choice == "Upload Image":
    image_file = st.file_uploader("Choose an image...")
    if image_file is not None:
        image = Image.open(image_file)
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        faces, labels, annotated_img = dnn_extract_face(img_array)
        if not faces:
            st.warning("No face detected.")
        else:
            st.image(annotated_img, caption="Detected Faces", width=400)
            for i, face in enumerate(faces):
                face_resized = cv2.resize(face, (350, 350))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_array = np.expand_dims(np.array(Image.fromarray(face_rgb)), axis=0)
                pred = model1.predict(face_array, verbose=0)
                pred = np.squeeze(pred)
                predIndex = np.argmax(pred)
                st.markdown(f"**Face {i+1}: {faces_db[predIndex]} (Accuracy {pred[predIndex]*100:.2f}%) - {labels[i]}**")

# -------------------------------
# Take a Snapshot
# -------------------------------
if choice == "Take a Snapshot":
    class SnapshotProcessor(VideoProcessorBase):
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]

        def __init__(self):
            self.frame_lock = threading.Lock()
            self.in_image = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            with self.frame_lock:
                self.in_image = img
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(key="snapshot",
                          mode=WebRtcMode.SENDRECV,
                          rtc_configuration=RTC_CONFIGURATION,
                          video_processor_factory=SnapshotProcessor,
                          media_stream_constraints={"video": True, "audio": False},
                          async_processing=True)

    if ctx.video_processor:
        if st.button("Capture Snapshot"):
            with ctx.video_processor.frame_lock:
                frame = ctx.video_processor.in_image
            if frame is not None:
                faces, labels, annotated_img = dnn_extract_face(frame)
                st.image(annotated_img, caption="Detected Faces", width=400)
                for i, face in enumerate(faces):
                    face_resized = cv2.resize(face, (350, 350))
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_array = np.expand_dims(np.array(Image.fromarray(face_rgb)), axis=0)
                    pred = model1.predict(face_array, verbose=0)
                    pred = np.squeeze(pred)
                    predIndex = np.argmax(pred)
                    st.markdown(f"**Face {i+1}: {faces_db[predIndex]} (Accuracy {pred[predIndex]*100:.2f}%) - {labels[i]}**")
            else:
                st.warning("No snapshot available. Please start the camera.")

# -------------------------------
# Real-Time Detection
# -------------------------------
if choice == "Real Time Detection":
    st.markdown('<h2 align="center">Real Time Masked Face Recognition</h2>', unsafe_allow_html=True)

    class Detection(NamedTuple):
        Name: str
        Prob: float

    class RealTimeProcessor(VideoProcessorBase):
        def __init__(self):
            self.result_queue = queue.Queue()
            self.frame_count = 0
            self.prev_time = time.time()
            self.fps = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)

            self.frame_count += 1
            # Resize frame for faster detection
            small_frame = cv2.resize(img, (640, 480))

            if self.frame_count % 5 == 0:  # skip frames
                faces_detected, labels, annotated_img = dnn_extract_face(small_frame)
                results = []
                for face in faces_detected:
                    face_resized = cv2.resize(face, (350, 350))
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    face_array = np.expand_dims(np.array(Image.fromarray(face_rgb)), axis=0)
                    pred = model1.predict(face_array, verbose=0)
                    pred = np.squeeze(pred)
                    predIndex = np.argmax(pred)
                    if pred[predIndex] > 0.95:
                        results.append(Detection(Name=faces_db[predIndex], Prob=float(pred[predIndex])))
                if results:
                    self.result_queue.put(results)

            # Calculate FPS
            current_time = time.time()
            self.fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            cv2.putText(img, f"FPS: {self.fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=RealTimeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    labels_placeholder = st.empty()
    if st.checkbox("Show detected faces table", value=True):
        while webrtc_ctx.state.playing:
            if webrtc_ctx.video_processor:
                try:
                    result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)

# -------------------------------
# About section
# -------------------------------
if choice == "About":
    st.markdown("""
        ### About
        This is a Masked Face Recognition Application.
        - Detect your face by Uploading an Image or by Taking a Snapshot.
        - Detect your face in real-time video.
        - Detect if you are wearing a mask or not.
        - Recognizes multiple faces at a time.
    """)
