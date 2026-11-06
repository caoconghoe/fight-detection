"""Lớp cơ sở cho mọi detector (CLAUDE.md mục 5.3)."""
from core.events import PRIORITY


class Detector:
    name = "base"

    def __init__(self, fps=15, **params):
        self.fps = fps                                    # dữ liệu vào được lấy mẫu ở fps này (15)

    def update(self, frame, tracks, t):
        """Gọi ở MỖI frame được lấy mẫu (15 FPS – khi sampler.tick(t) == True).
        frame: ảnh BGR; tracks: {id: {"box", "kpts"}}; t: thời điểm (giây).
        Trả về list sự kiện vừa XÁC NHẬN (không trả lặp mỗi frame)."""
        return []

    def highlight_ids(self):
        """(tuỳ chọn) ID cần tô đỏ trên màn hình, vd cặp đang đánh nhau."""
        return set()

    @staticmethod
    def event(type_, ids, confidence=None):
        return {"type": type_, "severity": PRIORITY[type_],
                "ids": sorted(int(i) for i in ids), "confidence": confidence}
