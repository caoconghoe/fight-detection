"""YOLOv8n-pose (OpenVINO) + ByteTrack -> dict tracks theo CLAUDE.md mục 5.1."""
import numpy as np
from ultralytics import YOLO


def dedup(tracks, iou_thr=0.6, contain_thr=0.8):
    """Loại phát hiện trùng: 2 khung có IoU > iou_thr hoặc phần giao > contain_thr diện tích
    khung nhỏ hơn (khung nằm trong khung kia) -> giữ khung có tổng conf keypoint cao hơn."""
    if len(tracks) < 2:
        return tracks
    order = sorted(tracks, key=lambda tid: -float(tracks[tid]["kpts"][:, 2].sum()))
    kept = []
    for tid in order:
        a = tracks[tid]["box"]
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        dup = False
        for k in kept:
            b = tracks[k]["box"]
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            inter = max(0.0, iw) * max(0.0, ih)
            iou = inter / (area_a + area_b - inter + 1e-9)
            contain = inter / (min(area_a, area_b) + 1e-9)
            if iou > iou_thr or contain > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(tid)
    return {tid: tracks[tid] for tid in kept}


class PoseTracker:
    def __init__(self, model, imgsz=416, conf=0.3, tracker="bytetrack.yaml", device="cpu",
                 dedup_iou=0.6, dedup_contain=0.8):
        self.model = YOLO(str(model), task="pose")
        self.imgsz, self.conf, self.tracker, self.device = imgsz, conf, tracker, device
        self.dedup_iou, self.dedup_contain = dedup_iou, dedup_contain

    def __call__(self, frame):
        """tracks = {track_id: {"box": (4,) xyxy pixel, "kpts": (17, 3) [x, y, conf]}}"""
        r = self.model.track(frame, imgsz=self.imgsz, conf=self.conf, tracker=self.tracker,
                             persist=True, device=self.device, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return {}
        ids = r.boxes.id.int().cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        kpts = r.keypoints.data.cpu().numpy().astype(np.float32)
        tracks = {int(tid): {"box": boxes[i], "kpts": kpts[i]} for i, tid in enumerate(ids)}
        return dedup(tracks, self.dedup_iou, self.dedup_contain)
