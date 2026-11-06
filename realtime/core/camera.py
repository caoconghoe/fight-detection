"""Đọc khung hình từ webcam, camera điện thoại (IP Webcam) hoặc file video.

- Nguồn trực tiếp (webcam / URL): luồng nền đọc liên tục, chỉ giữ frame MỚI NHẤT
  để không bị trễ dồn khi xử lý chậm hơn camera. t = time.time().
- File video: đọc tuần tự từng frame, t = chỉ số frame / FPS của video
  (để lấy mẫu 15 FPS đúng theo thời gian của video). Có thể phát đúng tốc độ thật.
"""
import threading
import time
from pathlib import Path

import cv2


BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}


class FpsMeter:
    """Đếm số sự kiện mỗi giây (cập nhật sau mỗi ≥ 1 s)."""
    def __init__(self):
        self.n, self.t0, self.value = 0, time.perf_counter(), 0.0

    def tick(self):
        self.n += 1
        dt = time.perf_counter() - self.t0
        if dt >= 1.0:
            self.value, self.n, self.t0 = self.n / dt, 0, time.perf_counter()


class Camera:
    def __init__(self, source=0, width=None, height=None, fps=30, backend="msmf",
                 file_realtime=True, fail_sec=3.0):
        src = str(source)
        self.src = int(src) if src.isdigit() else src
        self.is_file = isinstance(self.src, str) and Path(self.src).is_file()
        self.name = Path(src).name if self.is_file else src
        self.file_realtime, self.fail_sec = file_realtime, fail_sec
        self.meter = FpsMeter()                              # số frame camera thực sự trả về mỗi giây

        if isinstance(self.src, int):
            # webcam: MSMF cho ổn định 30 FPS; DSHOW mặc định có thể tụt 15 FPS khi thiếu sáng
            order = [backend] + [b for b in ("msmf", "dshow", "any") if b != backend]
            for b in order:
                self.cap = cv2.VideoCapture(self.src, BACKENDS[b])
                if self.cap.isOpened():
                    self.backend = b
                    break
        else:
            self.cap = cv2.VideoCapture(self.src)
            self.backend = self.cap.getBackendName() if self.cap.isOpened() else "?"
        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon video: {source}")
        if not self.is_file:
            if width and height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = f if 1 <= f <= 120 else 30.0

        self.idx = 0
        self.wall0 = None
        if not self.is_file:
            self.frame, self.t, self.seq, self.read_seq = None, None, 0, 0
            self.running = True
            self.cond = threading.Condition()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        last_ok = time.time()
        while self.running:
            ok, frame = self.cap.read()
            now = time.time()
            if not ok:
                if now - last_ok > self.fail_sec:            # mất tín hiệu quá lâu -> dừng
                    break
                time.sleep(0.01)
                continue
            last_ok = now
            self.meter.tick()
            with self.cond:
                self.frame, self.t, self.seq = frame, now, self.seq + 1
                self.cond.notify_all()
        with self.cond:
            self.running = False
            self.cond.notify_all()

    def read(self):
        """Trả về (frame BGR, t giây) hoặc (None, None) khi hết video / mất tín hiệu."""
        if self.is_file:
            ok, frame = self.cap.read()
            if not ok:
                return None, None
            t = self.idx / self.fps
            self.idx += 1
            self.meter.tick()
            if self.file_realtime:                            # phát đúng tốc độ video
                if self.wall0 is None:
                    self.wall0 = time.perf_counter()
                wait = self.wall0 + t - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -0.5:                             # trễ nhiều (vd lúc khởi động) -> neo lại, không chạy dồn
                    self.wall0 = time.perf_counter() - t
            return frame, t
        with self.cond:                                       # chờ frame mới (không xử lý lặp 1 frame)
            while self.running and self.seq == self.read_seq:
                self.cond.wait(timeout=0.5)
            if self.seq == self.read_seq:
                return None, None
            self.read_seq = self.seq
            return self.frame, self.t

    def release(self):
        if not self.is_file:
            self.running = False
            self.thread.join(timeout=1.0)
        self.cap.release()
