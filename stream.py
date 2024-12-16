import yt_dlp
import cv2
import numpy as np
from typing import Dict, List
import time
from yolo import YOLOProcessor, DetectionResult

class FrameRateController:
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
    CLASS_SELECTION = {
        0: 'person', 1: 'bicycle', 2: 'car',
        3: 'motorcycle', 5: 'bus', 7: 'truck'
    }

    def __init__(self, url: str, model_path: str, process_interval: int = 10):
        self.url = url
        self.process_interval = process_interval
        self.frame_counter = 0
        self.current_detections: List[DetectionResult] = []
        self.yolo = YOLOProcessor(model_path)
        self.fps_controller = FrameRateController(30)
        self.color_mapping = self._create_color_mapping()

    def _create_color_mapping(self) -> Dict[str, tuple]:
        colors = [
            (0, 255, 255), (255, 0, 0), (0, 255, 0),
            (0, 0, 255), (255, 0, 255), (255, 128, 0)
        ]
        return {class_name: colors[i] for i, (_, class_name)
                in enumerate(self.CLASS_SELECTION.items())}

    def _get_video_url(self) -> str:
        with yt_dlp.YoutubeDL({'format': 'best', 'quiet': True}) as ydl:
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

    def run(self):
        video_url = self._get_video_url()
        cap = cv2.VideoCapture(video_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)


        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if self.frame_counter == 0:
                    self.current_detections = self.yolo.process_frame(
                        frame, list(self.CLASS_SELECTION.keys())
                    )

                draw_frame = self._draw_detections(frame)
                self.frame_counter = (self.frame_counter + 1) % self.process_interval

                self.fps_controller.wait()
                cv2.imshow('Live Stream', draw_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    YOUTUBE_URL = "https://www.youtube.com/watch?v=rnXIjl_Rzy4"
    processor = StreamProcessor(YOUTUBE_URL, "yolo11x.pt")
    processor.run()
