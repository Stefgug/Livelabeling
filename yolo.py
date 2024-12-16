from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Dict, Tuple, Type

@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    class_id: int
    confidence: float
    label: str

class YOLOProcessor:
    def __init__(self, model_path: str, confidence: float = 0.6):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def process_frame(self, frame, class_selection: List[int]) -> List[Type[DetectionResult]]:
        results = self.model(
            frame,
            classes=class_selection,
            conf=self.confidence,
            device='cuda:0',
            half=True,
            max_det=100,
            imgsz=(1088,1920)

        )

        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                detections.append(DetectionResult(
                    bbox=tuple(map(int, box.xyxy[0])),
                    class_id=int(box.cls[0]),
                    confidence=float(box.conf[0]),
                    label=result.names[int(box.cls[0])]
                ))

        return detections
