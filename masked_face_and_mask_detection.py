import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration, WebRtcMode
from PIL import Image
import av
import queue
from typing import List, NamedTuple
from streamlit_option_menu import option_menu
import threading
from typing import Union


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.set_page_config(
    page_title="Masked-Face-Recognition",
    page_icon="😷",
    layout="wide"
)
load_css('css/styles.css')


@st.cache(allow_output_mutation=True)
def dnn_extract_face(img):
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    (height, width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (350, 350)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    # detections = np.squeeze(net.forward())
    face = None
    predition = None
    predIndex = None
    face_list = []
    label_list = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
#         print("confidence ",confidence)
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            # text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 0), 2)
            # cv2.putText(img, text, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            face = img[startY:endY, startX:endX]

            img2 = cv2.resize(face, (350, 350))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img2, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            preds = model2.predict(img_array)
            for pred in preds:
                (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 235, 0)
            label = "{}: {:.1f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(img, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            # cv2.rectangle(img, (startX, endY + 5), (endX, endY), color, cv2.FILLED)
            # return face
            face_list.append(face)
            label_list.append(label)
            #
            # if choice == "Real Time Detection":
            #     if type(face) is np.ndarray:
            #         face_new = cv2.resize(face, (350, 350))
            #         face_new = cv2.cvtColor(face_new, cv2.COLOR_BGR2RGB)
            #         im = Image.fromarray(face_new, 'RGB')
            #         img_array = np.array(im)
            #         img_array = np.expand_dims(img_array, axis=0)
            #         pred = model1.predict(img_array)
            #         predition = np.squeeze(pred)
            #         predIndex = np.argmax(predition)
            #
            #         # name = 'None matching'
            #         if (predition[predIndex] > 0.95):
            #             text = "{:.2f}%".format(predition[predIndex] * 100)
            #             name = str(faces[predIndex]) + ' ' + str(text)
            #             # cv2.putText(img, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            #             cv2.putText(img, name, (startX - 20, endY + 22), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
            #         else:
            #             cv2.putText(img, '', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                # else:
                #     cv2.putText(img, 'No Face Detected :(', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                #             cv2.putText(frame,'',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        # else:
        #     return None
    return face_list, face, label_list, predition, predIndex


@st.cache(allow_output_mutation=True)
def load_model():
    model1 = tf.keras.models.load_model('masked_face_detector.h5')
    model2 = tf.keras.models.load_model('mask_detector.h5')
    return model1, model2

# app = Flask(__name__)

model1, model2 = load_model()

# hide_streamlit_style = """
#     <style>
#     ul[data-testid=main-menu-list] > li:nth-of-type(2), /* Settings */
#     ul[data-testid=main-menu-list] > li:nth-of-type(3), /* Record a screencast */
#     ul[data-testid=main-menu-list] > li:nth-of-type(4), /* Report a bug */
#     ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Get help */
#     ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Share this app */
#     ul[data-testid=main-menu-list] > li:nth-of-type(7), /* About */
#     ul[data-testid=main-menu-list] > li:nth-of-type(8),
#     ul[data-testid=main-menu-list] > li:nth-of-type(9),
#     ul[data-testid=main-menu-list] > li:nth-of-type(10),
#     ul[data-testid=main-menu-list] > div:nth-of-type(2), /* 2nd divider */
#     ul[data-testid=main-menu-list] > div:nth-of-type(3),
#     ul[data-testid=main-menu-list] > div:nth-of-type(4),
#     ul[data-testid=main-menu-list] > div:nth-of-type(5),
#     ul[data-testid=main-menu-list] > div:nth-of-type(6)
#     {display: none;}
#     </style>
# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

faces = ['Abdur Samad', 'Ahsan Ahmed', 'Asef', 'Ashik', 'Azizul Hakim', 'DDS', 'Mahmud', 'Mayaz', 'Meheraj',
         'Nayeem Khan', 'Nayem', 'Risul Islam Fahim', 'Saif', 'Saki', 'Samir', 'Shahtab', 'Shimul Rahman Fahad',
         'Shourov', 'Shuvo', 'Sizan']

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]}
# )

choice = option_menu("Masked Face Recognition App", ["Upload Image", "Take Snapshot", "Real Time Detection", "About"],
                     icons=['file-earmark-arrow-up', 'camera', 'camera-video', 'house'],
                     menu_icon="emoji-smile", default_index=0, orientation="horizontal",
                     styles={
                         "container": {"background-color": "#002522"},
                         "icon": {"color": "orange", "font-size": "25px"},
                         "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                      "--hover-color": "#ff2d00"},
                         "nav-link-selected": {"background-color": "#02ab21"},
                        }
                     )

if choice == "Upload Image":
    # st.subheader("Image")
    image_file = st.file_uploader("Choose an image...")

    if image_file is not None:
        image = Image.open(image_file)
        i = 1

        # col1, col2 = st.columns([0.2, 0.5])
        # with col1:
        st.image(image_file, width=250, caption='Uploaded Image.')
        img_array = np.array(image)
        img_array, x, label, y, z = dnn_extract_face(img_array)
        if not img_array:
            st.warning("No face is detected.")
        else:
            for images in img_array:
                img_array = Image.fromarray(images)
                imResize = img_array.resize((350, 350), Image.ANTIALIAS)
                # imResize.save('predict.jpg', 'JPEG', quality=90)

                predictimg = np.array(imResize)
                predictimg = predictimg / 255.0
                predictimg = np.expand_dims(predictimg, axis=0)
                predition = model1.predict(predictimg)
                predition = np.squeeze(predition)
                predIndex = np.argmax(predition)
                st.markdown(""" <style> .font {
                    font-size: 46px; font-family: ''; color: white;} 
                    </style> """, unsafe_allow_html=True)
                # with col2:
                st.image(imResize, width=200, caption='Extracted Face '+str(i))
                st.markdown('<p class="font"><b>You are %s (Accuracy %.2f%%)</b></p>' % (
                faces[predIndex], predition[predIndex] * 100), unsafe_allow_html=True)

                # preds = model2.predict(predictimg)
                # for pred in preds:
                #     (mask, withoutMask) = pred
                # label = "a mask" if mask > withoutMask else "no mask"
                st.markdown('<p class="font"><b>Wearing %s</b></p>' % (label[i-1]), unsafe_allow_html=True)
                st.markdown('-----------------------------------------------', unsafe_allow_html=True)
                i += 1


if choice == "Take Snapshot":
    class VideoProcessor2(VideoProcessorBase):
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None

        def recv(self, frame: av.VideoFrame) -> np.ndarray:
            frame = frame.to_ndarray(format="bgr24")
            in_image = cv2.flip(frame, 1)

            with self.frame_lock:
                self.in_image = in_image
            return av.VideoFrame.from_ndarray(frame, format="bgr24")


    ctx = webrtc_streamer(key="snapshot",
                          mode=WebRtcMode.SENDRECV,
                          rtc_configuration=RTC_CONFIGURATION,
                          video_processor_factory=VideoProcessor2,
                          # media_stream_constraints={"video": True, "audio": False},
                          media_stream_constraints={
                              "video": {"width": 400, "ideal": 1200, "max": 1920},
                              "audio": False
                          },
                          async_processing=True
                          )

    if ctx.video_processor:
        if st.button("Snapshot"):
            with ctx.video_processor.frame_lock:
                in_image = ctx.video_processor.in_image

            if in_image is not None:
                image = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
                i = 1
                # st.write("Input image:")
                # st.image(in_image, channels="BGR")
                # st.write("Output image:")
                # st.image(out_image, channels="BGR")
                # col1, col2 = st.columns([0.3, 0.5])
                # with col1:
                st.image(image, width=400, caption='Snapshot Image.')
                img_array = np.array(image)
                img_array, x, label, y, z = dnn_extract_face(img_array)
                if not img_array:
                    st.warning("No face is detected.")
                else:
                    for images in img_array:
                        img_array = Image.fromarray(images)
                        imResize = img_array.resize((350, 350), Image.ANTIALIAS)
                        # imResize.save('predict.jpg', 'JPEG', quality=90)

                        predictimg = np.array(imResize)
                        predictimg = predictimg / 255.0
                        predictimg = np.expand_dims(predictimg, axis=0)

                        predition = model1.predict(predictimg)
                        predition = np.squeeze(predition)
                        predIndex = np.argmax(predition)
                        st.markdown(""" <style> .font {
                                        font-size: 46px; font-family: ''; color: white;} 
                                        </style> """, unsafe_allow_html=True)
                        # with col2:
                        st.image(imResize, width=200, caption='Extracted Face '+str(i))
                        st.markdown('<p class="font"><b>You are %s (Accuracy %.2f%%)</b></p>' % (
                        faces[predIndex], predition[predIndex] * 100), unsafe_allow_html=True)

                        # preds = model2.predict(predictimg)
                        # for pred in preds:
                        #     (mask, withoutMask) = pred
                        # label = "a mask" if mask > withoutMask else "no mask"
                        st.markdown('<p class="font"><b>Wearing %s</b></p>' % (label[i-1]), unsafe_allow_html=True)
                        st.markdown('-----------------------------------------------', unsafe_allow_html=True)
                        i += 1
            else:
                st.warning("No snapshot available yet. Please take a snapshot.")


if choice == "About":
    st.markdown(""" <style> .font {
    font-size: 40px; font-family: ''; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><b>About</b></p>', unsafe_allow_html=True)
    st.write(
        "This is a Masked Face Recognition Application. You can detect your face by Uploading an Image or by Taking a Snapshot. You can also detect your face in real time by using the Real Time Detection. This application can also detect if you are wearing a mask or not. This system can detect and recognize multiple faces at a time.")

if choice == "Real Time Detection":
    st.markdown('<h2 align="center">Real Time Masked Face Recognition</h2>', unsafe_allow_html=True)


    @st.cache(allow_output_mutation=True)
    class Detection(NamedTuple):
        Name: str
        Prob: float


    @st.cache(allow_output_mutation=True)
    class VideoProcessor(VideoProcessorBase):
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            result: List[Detection] = []
            frame = frame.to_ndarray(format="bgr24")
            frame = cv2.flip(frame,1)
            # frame = frame[:, ::-1, :]
            x, face, l, predition, predIndex = dnn_extract_face(frame)
            if type(face) is np.ndarray:
                face = cv2.resize(face, (350, 350))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(face, 'RGB')
                img_array = np.array(im)
                img_array = np.expand_dims(img_array, axis=0)
                pred = model1.predict(img_array)
                predition = np.squeeze(pred)
                predIndex = np.argmax(predition)

                #             name = 'None matching'
                if (predition[predIndex] > 0.95):
                    text = "{:.2f}%".format(predition[predIndex] * 100)
                    name = str(faces[predIndex]) + ' ' + str(text)
                    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
                    result.append(Detection(Name=faces[predIndex], Prob=float(predition[predIndex])))
                else:
                    cv2.putText(frame, '', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, 'No Face Detected :(', (30, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            #             cv2.putText(frame,'',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            self.result_queue.put(result)
            return av.VideoFrame.from_ndarray(frame, format="bgr24")


    webrtc_ctx = webrtc_streamer(key="key",
                                 mode=WebRtcMode.SENDRECV,
                                 rtc_configuration=RTC_CONFIGURATION,
                                 video_processor_factory=VideoProcessor,
                                 # media_stream_constraints={"video": True, "audio": False},
                                 media_stream_constraints={
                                     "video": {"width": 400, "ideal": 1200, "max": 1920},
                                     "audio": False
                                 },
                                 async_processing=True
                                 )

    st.markdown("""
                <style>
                table td:nth-child(1) {
                    display: none
                }
                table th:nth-child(1) {
                    display: none
                }
                table th {
                    text-align: center !important;
                    font-size: 130% !important;           
                }
                table td {
                    text-align: center !important;
                    color: lime !important;
                    font-size: 130% !important;
                }
                </style>
                """, unsafe_allow_html=True)
    if st.checkbox("Show the detected face", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break


# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/')
# def index():
#     return render_template('index.html')


# if __name__ == '__main__':
#    app.run(debug=True)



