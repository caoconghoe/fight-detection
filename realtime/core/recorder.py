"""Ghi clip .mp4 quanh sự kiện (CLAUDE.md A5).

Giữ buffer_sec giây khung hình ĐÃ VẼ gần nhất (ring buffer, lấy mẫu `fps` theo thời gian, thu nhỏ nếu rộng
hơn max_width để giới hạn bộ nhớ). Khi có sự kiện: lấy pre_sec giây trước trong buffer + gom tiếp post_sec giây
sau, rồi ghi .mp4 (codec mp4v) ở LUỒNG NỀN; xong gọi on_done(event_id, path) (vd db.set_clip).
"""
import threading
from collections import deque
from pathlib import Path

import cv2

from shared.sampling import FpsSampler


class ClipRecorder:
    def __init__(self, out_dir, fps=15, buffer_sec=5.0, pre_sec=3.0, post_sec=2.0, max_width=1280,
                 codec="mp4v", on_done=None):
        self.out_dir = Path(out_dir)
        self.fps, self.pre, self.post = fps, pre_sec, post_sec
        self.max_width, self.codec, self.on_done = max_width, codec, on_done
        self.buf = deque(maxlen=int(buffer_sec * fps) + 1)          # (t, frame)
        self.sampler = FpsSampler(fps)
        self.jobs = []                                              # clip đang gom khung hình sau sự kiện
        self.threads = []

    def push(self, frame, t):
        """Gọi mỗi frame với ảnh ĐÃ VẼ (không bị sửa sau đó) và thời điểm t (giây)."""
        if not self.sampler.tick(t):
            return
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        self.buf.append((t, frame))
        for job in self.jobs:
            job["frames"].append((t, frame))
        for job in [j for j in self.jobs if t >= j["t_end"]]:
            self.jobs.remove(job)
            self._write_async(job)

    def trigger(self, event_id, name, t):
        """Bắt đầu clip cho sự kiện tại thời điểm t (gọi SAU push của frame hiện tại). name: tên file không dấu."""
        frames = [(ts, f) for ts, f in self.buf if ts >= t - self.pre]
        self.jobs.append({"event_id": event_id, "path": self.out_dir / f"{name}.mp4",
                          "t_end": t + self.post, "frames": frames})

    def _write_async(self, job):
        th = threading.Thread(target=self._write, args=(job,), daemon=False)
        th.start()
        self.threads = [x for x in self.threads if x.is_alive()] + [th]

    def _write(self, job):
        frames = job["frames"]
        if not frames:
            print("[recorder] khong co khung hinh cho su kien", job["event_id"])
            return
        span = frames[-1][0] - frames[0][0]
        fps = min(self.fps, max(1.0, (len(frames) - 1) / span)) if span > 0 else self.fps   # FPS thật
        h, w = frames[0][1].shape[:2]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = job["path"]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.codec), fps, (w, h))
        if not vw.isOpened():
            print("[recorder] LOI mo VideoWriter:", path)
            return
        for _, f in frames:
            vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        vw.release()
        if path.is_file() and path.stat().st_size > 0:
            if self.on_done:
                self.on_done(job["event_id"], path)
        else:
            print("[recorder] LOI ghi clip:", path)

    def close(self):
        """Thoát chương trình: ghi nốt các clip đang gom dở (ngắn hơn post_sec) và chờ mọi luồng ghi xong."""
        for job in self.jobs:
            self._write_async(job)
        self.jobs = []
        for th in self.threads:
            th.join()
        self.threads = []
