"""TÉ NGÃ: hông rơi nhanh rồi thân nằm ngang liên tục ≥ hold_sec (CLAUDE.md mục 9.1)."""
from collections import deque

import numpy as np

from detectors.base import Detector

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
CONF = 0.4


class FallDetector(Detector):
    name = "fall"

    def __init__(self, fps=15, drop_speed=1.2, angle_deg=60, hold_sec=2.0, drop_grace_sec=1.0, **kw):
        super().__init__(fps, **kw)
        self.drop_speed = drop_speed                      # số "thân người" (vai→hông) hông rơi trong 1 s
        self.angle, self.hold = angle_deg, int(hold_sec * fps)
        self.grace = drop_grace_sec                       # sau cú rơi, chờ tối đa N giây để thân nằm ngang
        self.hist, self.dropped, self.down_count, self.fired = {}, {}, {}, set()

    def _check(self, tid, kpts, box, t):
        if min(kpts[[L_SH, R_SH, L_HIP, R_HIP], 2]) < CONF:
            return False
        sh = kpts[[L_SH, R_SH], :2].mean(0)
        hip = kpts[[L_HIP, R_HIP], :2].mean(0)
        dx, dy = hip - sh
        angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))   # 0° = thân thẳng đứng
        torso = float(np.hypot(dx, dy))
        w, h = box[2] - box[0], box[3] - box[1]
        q = self.hist.setdefault(tid, deque(maxlen=self.fps))
        q.append((t, hip[1], torso))
        if len(q) == q.maxlen:
            body = np.median([b for _, _, b in q])
            speed = (q[-1][1] - q[0][1]) / max(body, 1.0) / max(q[-1][0] - q[0][0], 1e-6)  # >0: hông đi xuống
            if speed > self.drop_speed:
                self.dropped[tid] = t
        lying = angle > self.angle or w > 1.2 * h
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
            for d in (self.hist, self.dropped, self.down_count):
                d.pop(tid, None)
            self.fired.discard(tid)
        return [self.event("TÉ NGÃ", [tid]) for tid, tr in tracks.items()
                if self._check(tid, tr["kpts"], tr["box"], t)]
