"""TÉ NGÃ: hông rơi nhanh rồi thân nằm ngang liên tục ≥ hold_sec (CLAUDE.md mục 9.1).
Chỉ xét khi thấy ĐỦ THÂN (Detector.body_check) – tránh báo nhầm khi ngồi gần camera / cúi người."""
from collections import deque

import numpy as np

from detectors.base import Detector

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


class FallDetector(Detector):
    name = "fall"

    def __init__(self, fps=15, drop_speed=1.2, angle_deg=60, hold_sec=2.0, drop_grace_sec=1.0,
                 scale_sec=2.0, **kw):
        super().__init__(fps, **kw)
        self.drop_speed = drop_speed                      # số "thân người" (vai→hông) hông rơi trong 1 s
        self.angle, self.hold = angle_deg, int(hold_sec * fps)
        self.grace = drop_grace_sec                       # sau cú rơi, chờ tối đa N giây để thân nằm ngang
        self.scale_sec = scale_sec                        # thang đo = trung vị chiều dài thân N giây TRƯỚC cửa sổ rơi
        self.hist, self.dropped, self.down_count, self.fired = {}, {}, {}, set()
        self.dbg = {}                                     # tid -> thông tin cho chế độ debug

    def _speed(self, q, t):
        """Tốc độ rơi của hông trong ~1 s gần nhất, chia cho chiều dài thân ĐO TRƯỚC cửa sổ đó
        (cúi người làm thân ngắn lại -> nếu dùng giá trị hiện tại sẽ phóng đại tốc độ)."""
        win = [e for e in q if t - e[0] <= 1.0 + 1e-6]
        t0, hip0, _ = win[0]
        pre = [e[2] for e in q if t0 - self.scale_sec <= e[0] < t0]
        if t - t0 < 0.6 or len(pre) < self.fps // 2:
            return None
        return (q[-1][1] - hip0) / max(float(np.median(pre)), 1.0) / (t - t0)   # >0: hông đi xuống

    def _check(self, tid, kpts, box, t, shape):
        full, touch, miss = self.body_check(kpts, box, shape)
        if not full:
            self.dbg[tid] = {"miss": miss}
            return False
        sh = kpts[[L_SH, R_SH], :2].mean(0)
        hip = kpts[[L_HIP, R_HIP], :2].mean(0)
        dx, dy = hip - sh
        angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))   # 0° = thân thẳng đứng
        torso = float(np.hypot(dx, dy))
        w, h = box[2] - box[0], box[3] - box[1]
        q = self.hist.setdefault(tid, deque(maxlen=int((1 + self.scale_sec) * self.fps) + 2))
        q.append((t, hip[1], torso))
        speed = self._speed(q, t)
        if speed is not None and speed > self.drop_speed:
            self.dropped[tid] = t
        lying = angle > self.angle or (not touch and w > 1.2 * h)  # khung chạm mép: tỉ lệ w/h không tin được
        self.dbg[tid] = {"miss": [], "speed": speed, "angle": angle}
        drop_t = self.dropped.get(tid)
        if drop_t is not None and lying:
            self.down_count[tid] = self.down_count.get(tid, 0) + 1
        elif not lying:
            self.down_count[tid] = 0
            self.fired.discard(tid)                        # đứng dậy -> lần té sau được báo lại
            if drop_t is not None and t - drop_t > self.grace:
                del self.dropped[tid]
        if self.down_count.get(tid, 0) >= self.hold and tid not in self.fired:
            self.fired.add(tid)                            # chỉ báo 1 lần mỗi lần té
            return True
        return False

    def update(self, frame, tracks, t):
        for tid in self.gone_ids(tracks, t):
            for d in (self.hist, self.dropped, self.down_count, self.dbg):
                d.pop(tid, None)
            self.fired.discard(tid)
        shape = None if frame is None else frame.shape
        return [self.event("TÉ NGÃ", [tid]) for tid, tr in tracks.items()
                if self._check(tid, tr["kpts"], tr["box"], t, shape)]

    def debug_info(self, tid):
        d = self.dbg.get(tid)
        if d is None:
            return []
        if d["miss"]:
            return [f"toan than: KHONG ({', '.join(d['miss'])}) -> bo qua te/nam im"]
        sp = "-" if d["speed"] is None else f"{d['speed']:.2f}"
        state = " ROI" if tid in self.dropped else ""
        return ["toan than: co",
                f"TE roi {sp}/{self.drop_speed} goc {d['angle']:.0f}/{self.angle} "
                f"nam {self.down_count.get(tid, 0)}/{self.hold}{state}"]
