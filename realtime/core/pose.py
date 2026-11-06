"""YOLOv8n-pose (OpenVINO) + ByteTrack -> dict tracks theo CLAUDE.md mục 5.1."""
import numpy as np
from ultralytics import YOLO


class PoseTracker:
    def __init__(self, model, imgsz=416, conf=0.3, tracker="bytetrack.yaml", device="cpu"):
        self.model = YOLO(str(model), task="pose")
        self.imgsz, self.conf, self.tracker, self.device = imgsz, conf, tracker, device

    def __call__(self, frame):
        """tracks = {track_id: {"box": (4,) xyxy pixel, "kpts": (17, 3) [x, y, conf]}}"""
        r = self.model.track(frame, imgsz=self.imgsz, conf=self.conf, tracker=self.tracker,
                             persist=True, device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return {}
        ids = r.boxes.id.int().cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        kpts = r.keypoints.data.cpu().numpy().astype(np.float32)
        return {int(tid): {"box": boxes[i], "kpts": kpts[i]} for i, tid in enumerate(ids)}
