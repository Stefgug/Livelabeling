import streamlit as st
import requests
from PIL import Image
import io

def main():
    """
    Main Streamlit application for displaying real-time object detection.
    Connects to FastAPI backend and handles video stream display.
    """
    st.title("Live Object Detection")

    try:
        # Attempt to connect to the video feed
        response = requests.get("http://localhost:8000/video-feed", stream=True)
        if response.status_code == 404:
            st.error("Error: Video feed not found. Make sure the FastAPI server is running.")
            return

        headers = response.headers
        if 'content-type' not in headers:
            st.error("Error: Invalid response from server")
            return

        boundary = headers['content-type'].split('boundary=')[1]
        placeholder = st.empty()

        # Process multipart response and display frames
        for part in response.iter_content(chunk_size=10*1024*1024):
            if len(part) == 0:
                continue

            # Find the JPEG image data
            img_start = part.find(b'\xff\xd8')
            img_end = part.find(b'\xff\xd9')

            if img_start != -1 and img_end != -1:
                image = Image.open(io.BytesIO(part[img_start:img_end+2]))
                placeholder.image(image)

    except requests.exceptions.ConnectionError:
        st.error("Error: Could not connect to the video feed. Is the server running?")

if __name__ == "__main__":
    main()
