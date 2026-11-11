"""Lớp cơ sở cho mọi detector (CLAUDE.md mục 5.3)."""
from core.events import PRIORITY

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14
FULL_BODY = {"shoulder_hip_conf": 0.5, "knee_conf": 0.4, "edge_margin": 0.02}


class Detector:
    name = "base"

    def __init__(self, fps=15, forget_sec=1.0, full_body=None, **params):
        self.fps = fps                                    # dữ liệu vào được lấy mẫu ở fps này (15)
        self.forget_sec = forget_sec                      # ID vắng quá N giây mới xoá dữ liệu
        self.full_body = {**FULL_BODY, **(full_body or {})}
        self.last_seen = {}

    def body_check(self, kpts, box, shape):
        """Có thấy ĐỦ THÂN không (điều kiện để xét té ngã / nằm bất động):
        2 vai + 2 hông conf ≥ shoulder_hip_conf, ≥ 1 gối conf ≥ knee_conf, khung không chạm mép dưới ảnh.
        shape: kích thước ảnh (H, W, ...) hoặc None (bỏ qua kiểm tra mép).
        Trả về (đủ thân?, khung chạm mép bất kỳ?, [lý do thiếu, ASCII])."""
        c = self.full_body
        miss = []
        if min(kpts[[L_SH, R_SH, L_HIP, R_HIP], 2]) < c["shoulder_hip_conf"]:
            miss.append("vai/hong")
        if max(kpts[[L_KNEE, R_KNEE], 2]) < c["knee_conf"]:
            miss.append("goi")
        touch = False
        if shape is not None:
            H, W = shape[:2]
            mx, my = c["edge_margin"] * W, c["edge_margin"] * H
            bottom = box[3] >= H - my
            touch = bottom or box[1] <= my or box[0] <= mx or box[2] >= W - mx
            if bottom:
                miss.append("cham mep duoi")
        return not miss, touch, miss

    def update(self, frame, tracks, t):
        """Gọi ở MỖI frame được lấy mẫu (15 FPS – khi sampler.tick(t) == True).
        frame: ảnh BGR; tracks: {id: {"box", "kpts"}}; t: thời điểm (giây).
        Trả về list sự kiện vừa XÁC NHẬN (không trả lặp mỗi frame)."""
        return []

    def attach(self, detectors):
        """(tuỳ chọn) EventManager gọi 1 lần với danh sách mọi detector – để dùng kết quả của detector khác."""

    def highlight_ids(self):
        """(tuỳ chọn) ID cần tô đỏ trên màn hình, vd cặp đang đánh nhau."""
        return set()

    def debug_info(self, tid):
        """(chế độ debug, phím d) Các dòng chữ ASCII hiện trên khung người tid: bộ đếm hiện tại của luật."""
        return []

    def debug_scene(self):
        """(chế độ debug) Các dòng chữ cho cả cảnh (vd hoảng loạn), hiện trên panel."""
        return []

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
