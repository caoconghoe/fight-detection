"""Lớp cơ sở cho mọi detector (CLAUDE.md mục 5.3)."""
from core.events import PRIORITY


class Detector:
    name = "base"

    def __init__(self, fps=15, forget_sec=1.0, **params):
        self.fps = fps                                    # dữ liệu vào được lấy mẫu ở fps này (15)
        self.forget_sec = forget_sec                      # ID vắng quá N giây mới xoá dữ liệu
        self.last_seen = {}

    def update(self, frame, tracks, t):
        """Gọi ở MỖI frame được lấy mẫu (15 FPS – khi sampler.tick(t) == True).
        frame: ảnh BGR; tracks: {id: {"box", "kpts"}}; t: thời điểm (giây).
        Trả về list sự kiện vừa XÁC NHẬN (không trả lặp mỗi frame)."""
        return []

    def highlight_ids(self):
        """(tuỳ chọn) ID cần tô đỏ trên màn hình, vd cặp đang đánh nhau."""
        return set()

    def gone_ids(self, tracks, t):
        """Cập nhật thời điểm thấy từng ID; trả về các ID đã vắng > forget_sec (để detector xoá dữ liệu).
        Có thời gian chờ vì YOLO hay mất người (nhất là người đang nằm) trong vài frame."""
        for tid in tracks:
            self.last_seen[tid] = t
        gone = [tid for tid, ts in self.last_seen.items() if t - ts > self.forget_sec]
        for tid in gone:
            del self.last_seen[tid]
        return gone

    @staticmethod
    def event(type_, ids, confidence=None, **extra):
        return {"type": type_, "severity": PRIORITY[type_],
                "ids": sorted(int(i) for i in ids), "confidence": confidence, **extra}
