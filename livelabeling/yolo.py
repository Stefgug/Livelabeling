from ultralytics import YOLO
from typing import List, Dict, Tuple, Type


class DetectionResult:
    """Represents a single object detection result with its bounding box and metadata."""
    def __init__(self, bbox: Tuple[int, int, int, int], class_id: int, confidence: float, label: str):
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.label = label

class YOLOProcessor:
    """Handles object detection using the YOLO model with GPU acceleration."""
    def __init__(self, model_path: str, confidence: float = 0.6):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def process_frame(self, frame, class_selection: List[int]) -> List[Type[DetectionResult]]:
        """
        Process a single frame through YOLO detection.

        Args:
            frame: Input image frame
            class_selection: List of class IDs to detect

        Returns:
            List of DetectionResult objects
        """
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
