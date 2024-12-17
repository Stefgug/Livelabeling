import yt_dlp
import cv2
import numpy as np
from typing import Dict, List
import time
from livelabeling.yolo import YOLOProcessor, DetectionResult
import os


class FrameRateController:
    """Controls frame rate to maintain consistent video output speed."""
    def __init__(self, target_fps: int):
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = time.time()

    def wait(self) -> None:
        current_time = time.time()
        sleep_time = max(0, self.frame_interval - (current_time - self.last_frame_time))
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_frame_time = time.time()

class StreamProcessor:
    """
    Processes video streams with real-time object detection.
    Handles YouTube video ingestion and YOLO-based detection.
    """

    # Supported object classes for detection
    CLASS_SELECTION = {
        0: 'person', 1: 'bicycle', 2: 'car',
        3: 'motorcycle', 5: 'bus', 7: 'truck'
    }

    def __init__(self, url: str, model_path: str, process_interval: int = 3):
        """
        Initialize stream processor with video source and YOLO model.
        process_interval determines how often detection runs (every N frames)
        """
        self.url = url
        self.process_interval = process_interval
        self.frame_counter = 0
        self.current_detections: List[DetectionResult] = []
        self.yolo = YOLOProcessor(model_path)
        self.fps_controller = FrameRateController(25)
        self.color_mapping = self._create_color_mapping()
        self.last_pts = 0  # Add this line to track presentation timestamps

    def _create_color_mapping(self) -> Dict[str, tuple]:
        colors = [
            (0, 255, 255), (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (255, 0, 255), (255, 128, 0)
        ]
        return {class_name: colors[i] for i, (_, class_name)
                in enumerate(self.CLASS_SELECTION.items())}

    def _get_video_url(self) -> str:
        with yt_dlp.YoutubeDL({
            'format': 'best[ext=mp4]/best',  # Prefer MP4 format
            'quiet': True
        }) as ydl:
            info_dict = ydl.extract_info(self.url, download=False)
            return info_dict['url']

    def _draw_detections(self, frame) -> np.ndarray:
        draw_frame = frame.copy()
        for detection in self.current_detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.color_mapping[detection.label]
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_frame,
                       f"{detection.label} {detection.confidence:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        return draw_frame

    def process_frame(self):
        """
        Main processing loop that yields processed video frames.
        Manages video capture, object detection, and visualization.
        """
        video_url = self._get_video_url()
        cap = cv2.VideoCapture(video_url)

        # Improved capture settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Additional capture options to improve H264 decoding
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp,tcp,https,tls"

        frame_count = 0
        dropped_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if dropped_frames < 5:  # Try to recover a few times
                        dropped_frames += 1
                        continue
                    break

                frame_count += 1
                current_pts = cap.get(cv2.CAP_PROP_POS_MSEC)

                # Skip frame if we're falling behind, but avoid skipping keyframes
                if current_pts - self.last_pts > 500:  # More than 500ms behind
                    self.frame_counter = 0  # Force next frame to process
                    self.last_pts = current_pts
                    continue

                if self.frame_counter == 0:
                    try:
                        self.current_detections = self.yolo.process_frame(
                            frame, list(self.CLASS_SELECTION.keys())
                        )
                    except Exception as e:
                        print(f"Detection error: {e}")
                        self.current_detections = []

                draw_frame = self._draw_detections(frame)
                self.frame_counter = (self.frame_counter + 1) % self.process_interval
                self.last_pts = current_pts

                # Only control FPS if we're keeping up
                if frame_count % 30 == 0:  # Check every 30 frames
                    self.fps_controller.wait()

                _, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield buffer.tobytes()

                dropped_frames = 0  # Reset dropped frames counter on success

        except Exception as e:
            print(f"Stream error: {e}")

        finally:
            cap.release()

if __name__ == "__main__":
    YOUTUBE_URL = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"
    processor = StreamProcessor(YOUTUBE_URL, "models/yolo11x.pt")
    for frame in processor.process_frame():
        # Here you can handle the frame, e.g., send it to a web client
        pass
