from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from livelabeling.stream import StreamProcessor

app = FastAPI(
    title="Live Object Detection API",
    description="Serves real-time object detection from video streams"
)

# Configuration for demo stream
YOUTUBE_URL = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"
processor = StreamProcessor(YOUTUBE_URL, "models/yolo11x.pt")

def generate_frames():
    """Generator function for video streaming, yields JPEG frames with detection overlays."""
    for frame in processor.process_frame():
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/video-feed')
async def video_feed():
    """
    Stream processed video frames with object detection.
    Returns a multipart response containing JPEG frames.
    """
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
