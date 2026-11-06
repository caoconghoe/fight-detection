"""Lấy mẫu chuỗi frame về đúng FPS mục tiêu theo THỜI GIAN.

File này DÙNG CHUNG giữa realtime và notebook Kaggle -> chỉ dùng thư viện chuẩn.
Nguồn 30 FPS -> giữ ~1/2 số frame; nguồn 22 FPS (RLVS) -> giữ ~2/3; nguồn < fps -> giữ hết.
"""


class FpsSampler:
    def __init__(self, fps=15, tolerance=0.25):
        self.period = 1.0 / fps
        self.tol = tolerance * self.period     # bù rung thời gian (jitter) của camera
        self.next_t = None

    def reset(self):
        self.next_t = None

    def tick(self, t):
        """True nếu frame tại thời điểm t (giây) được lấy mẫu."""
        if self.next_t is None or t - self.next_t > self.period:   # frame đầu hoặc bị tụt lại xa
            self.next_t = t + self.period
            return True
        if t >= self.next_t - self.tol:
            self.next_t += self.period
            return True
        return False


def sample_indices(times, fps=15):
    """Offline: trả về chỉ số các frame được giữ, vd times = [i / fps_goc for i in range(n)]."""
    s = FpsSampler(fps)
    return [i for i, t in enumerate(times) if s.tick(t)]
