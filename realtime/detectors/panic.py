"""HOẢNG LOẠN (đám đông bỏ chạy) – luật tracking (CLAUDE.md mục 9.3).
Điều kiện cùng lúc ≥ hold_sec: ≥ min_people người; ≥ min_running người và ≥ run_ratio đang chạy
(tốc độ > run_speed chiều cao/giây); tốc độ TB cả cảnh > surge_ratio × mức nền (học dần khi bình thường)."""
from collections import deque

import numpy as np

from detectors.base import Detector


class PanicDetector(Detector):
    name = "panic"

    def __init__(self, fps=15, run_speed=1.5, min_people=4, min_running=3, run_ratio=0.5,
                 surge_ratio=2.0, hold_sec=1.0, cooldown_sec=10, **kw):
        super().__init__(fps, **kw)
        self.run_speed = run_speed
        self.min_people, self.min_running = min_people, min_running
        self.run_ratio, self.surge_ratio = run_ratio, surge_ratio
        self.hold, self.cooldown = int(hold_sec * fps), int(cooldown_sec * fps)
        self.tracks = {}
        self.baseline = None
        self.on_count, self.cool = 0, 0

    def _check(self, ids, boxes):
        seen = set()
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            q = self.tracks.setdefault(tid, deque(maxlen=self.fps))
            foot, h = ((x1 + x2) / 2, y2), (y2 - y1)
            if q and np.hypot(foot[0] - q[-1][0], foot[1] - q[-1][1]) > 1.5 * h:
                q.clear()                                   # nhảy vị trí bất thường (đổi ID) -> làm lại
            q.append((foot[0], foot[1], h))
            seen.add(tid)
        for tid in list(self.tracks):
            if tid not in seen:
                del self.tracks[tid]

        speeds, vels, starts = {}, {}, {}
        for tid, q in self.tracks.items():
            if len(q) < q.maxlen:
                continue
            arr = np.array(q)
            dt = (len(arr) - 1) / self.fps
            body = np.median(arr[:, 2])
            d = arr[-1, :2] - arr[0, :2]
            speeds[tid] = np.linalg.norm(d) / body / dt
            vels[tid], starts[tid] = d, arr[0, :2]

        if self.cool > 0:
            self.cool -= 1
        n = len(speeds)
        if n == 0:
            self.on_count = 0
            return None

        mean_speed = float(np.mean(list(speeds.values())))
        running = [t for t, s in speeds.items() if s > self.run_speed]
        if self.baseline is None:
            self.baseline = mean_speed
        surge = mean_speed > self.surge_ratio * max(self.baseline, 0.2)
        panic_now = (n >= self.min_people and len(running) >= self.min_running
                     and len(running) / n >= self.run_ratio and surge)
        if not panic_now:
            self.baseline = 0.99 * self.baseline + 0.01 * mean_speed
        self.on_count = self.on_count + 1 if panic_now else 0
        if self.on_count < self.hold or self.cool > 0:
            return None
        self.cool = self.cooldown

        center = np.mean([starts[t] for t in running], axis=0)
        outward = []
        for t in running:
            r, v = starts[t] - center, vels[t]
            if np.linalg.norm(r) > 1e-6 and np.linalg.norm(v) > 1e-6:
                outward.append(np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v)))
        dirs = np.array([vels[t] / (np.linalg.norm(vels[t]) + 1e-6) for t in running])
        if outward and np.mean(outward) > 0.5:
            kind = "tỏa ra nhiều hướng"
        elif np.linalg.norm(dirs.mean(0)) > 0.7:
            kind = "chạy cùng một hướng"
        else:
            kind = "chạy hỗn loạn"
        return dict(n_people=n, n_running=len(running), kind=kind, ids=running)

    def update(self, frame, tracks, t):
        """Phải gọi mọi tick, kể cả khi không có ai (để đếm cooldown/hold)."""
        r = self._check(list(tracks), [tr["box"] for tr in tracks.values()])
        if r is None:
            return []
        return [self.event("HOẢNG LOẠN", r["ids"], detail=f"{r['kind']}, {r['n_running']}/{r['n_people']} người chạy")]
