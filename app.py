import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import threading

st.title("YOLOv11 object detection on live YouTube video")

# Classes we're interested in
classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.youtube_url = None
        self.cap = None
        self.lock = threading.Lock()
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            try:
                self.model = YOLO('yolo11n.pt', verbose=False)
            except Exception as e:
                st.error(f"Error loading YOLO model: {str(e)}")
                return False
        return True

    def update_url(self, url):
        try:
            with self.lock:
                if self.cap is not None:
                    self.cap.release()
                ydl_opts = {'format': 'best'}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_url = info['url']
                    self.cap = cv2.VideoCapture(video_url)
                return True
        except Exception as e:
            st.error(f"Error updating URL: {str(e)}")
            return False

    def recv(self, frame):
        if not self._ensure_model():
            return frame

        img = frame.to_ndarray(format="bgr24")

        try:
            with self.lock:
                if self.cap and self.cap.isOpened():
                    ret, youtube_frame = self.cap.read()
                    if ret:
                        img = youtube_frame

            # Run detection
            results = self.model.predict(img, classes=list(classes.keys()), verbose=False)[0]

            # Draw detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                if class_id in classes:
                    label = f"{classes[class_id]} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        if self.cap:
            self.cap.release()

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = VideoProcessor()

# Input for YouTube URL
youtube_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=rnXIjl_Rzy4")

st.session_state.processor.update_url(youtube_url)

try:
    webrtc_streamer(
        key="example",
        video_processor_factory=lambda: st.session_state.processor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True
        },
    )
except Exception as e:
    st.error(f"Error initializing WebRTC stream: {str(e)}")
