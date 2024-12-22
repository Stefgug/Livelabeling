import streamlit as st
import cv2
import yt_dlp
from ultralytics import YOLO
import time

st.title("Live Youtube Object Detection")

def get_youtube_url(url):
    """Extract the direct video URL from YouTube link"""
    with yt_dlp.YoutubeDL({'format': 'best'}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def main():
    st.title("Live Youtube Object Detection")

    # Input for YouTube URL
    youtube_url = st.text_input("YouTube URL", "https://www.youtube.com/watch?v=rnXIjl_Rzy4")

    if not youtube_url:
        st.warning("Please enter a YouTube URL")
        return

    # Load YOLO model
    model = YOLO('yolov11n.pt')  # using smaller model for better performance

    # Classes we're interested in
    classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    # Setup video capture
    video_url = get_youtube_url(youtube_url)
    cap = cv2.VideoCapture(video_url)

    # Create placeholder for video feed
    frame_placeholder = st.empty()

    # Metrics
    fps_placeholder = st.empty()
    last_time = time.time()
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Can't receive frame from stream")
                break

            # Run detection
            results = model.predict(frame, classes=list(classes.keys()))[0]

            # Draw detections
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                if class_id in classes:
                    label = f"{classes[class_id]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame
            frame_placeholder.image(frame)

            # Update FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                fps_placeholder.text(f"FPS: {fps:.1f}")
                frame_count = 0
                last_time = time.time()

    finally:
        cap.release()

if __name__ == "__main__":
    main()
