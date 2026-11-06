"""Unit test core/recorder.py: clip đúng khoảng [t − 3 s, t + 2 s], ghi ở luồng nền, gọi on_done, thu nhỏ khung rộng.
Mỗi khung hình giả tô mức xám = 20 × t để đọc lại thời điểm từ clip.  Chạy: python realtime/tests/test_recorder.py"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.recorder import ClipRecorder  # noqa: E402

TMP = Path(tempfile.mkdtemp())


def frame_at(t, w=640, h=360):
    return np.full((h, w, 3), int(round(20 * t)) % 256, np.uint8)


def read_times(path):
    cap = cv2.VideoCapture(str(path))
    out, size = [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        size = f.shape[1::-1]
        out.append(f[50:-50, 50:-50].mean() / 20)
    return np.array(out), size, cap.get(cv2.CAP_PROP_FPS)


def feed(rec, t_end, fps_in=30.0, events=(), w=640):
    for i in range(int(t_end * fps_in)):
        t = i / fps_in
        rec.push(frame_at(t, w), t)
        for eid, te in events:
            if abs(t - te) < 0.5 / fps_in:
                rec.trigger(eid, f"su_kien_{eid}", t)


def test_clip_window_and_callback():
    done = {}
    rec = ClipRecorder(TMP / "a", on_done=lambda eid, p: done.__setitem__(eid, p))
    feed(rec, 10, events=[(7, 6.0)])
    rec.close()
    assert set(done) == {7} and done[7].name == "su_kien_7.mp4" and done[7].is_file()
    ts, size, fps = read_times(done[7])
    assert size == (640, 360) and abs(fps - 15) < 0.5, (size, fps)
    assert abs(ts[0] - 3.0) < 0.15 and abs(ts[-1] - 8.0) < 0.15, (ts[0], ts[-1])
    assert 72 <= len(ts) <= 78, len(ts)                                # 5 s × 15 FPS


def test_written_before_close_in_background():
    """Clip ghi ngay khi đủ 2 s sau sự kiện (không cần chờ close)."""
    done = {}
    rec = ClipRecorder(TMP / "b", on_done=lambda eid, p: done.__setitem__(eid, p))
    feed(rec, 9, events=[(1, 3.0)])
    for th in rec.threads:
        th.join()
    assert 1 in done
    rec.close()


def test_event_near_start_and_pending_at_close():
    """Sự kiện ở t = 1 s: chỉ có 1 s trước; sự kiện sát lúc thoát: close() ghi nốt phần đã có."""
    done = {}
    rec = ClipRecorder(TMP / "c", on_done=lambda eid, p: done.__setitem__(eid, p))
    feed(rec, 6, events=[(1, 1.0), (2, 5.5)])
    rec.close()
    t1, _, _ = read_times(done[1])
    t2, _, _ = read_times(done[2])
    assert abs(t1[0] - 0.0) < 0.15 and abs(t1[-1] - 3.0) < 0.15, (t1[0], t1[-1])
    assert abs(t2[0] - 2.5) < 0.15 and t2[-1] < 6.0, (t2[0], t2[-1])


def test_two_overlapping_events():
    done = {}
    rec = ClipRecorder(TMP / "d", on_done=lambda eid, p: done.__setitem__(eid, p))
    feed(rec, 10, events=[(1, 5.0), (2, 6.0)])
    rec.close()
    assert set(done) == {1, 2}
    assert abs(read_times(done[2])[0][0] - 3.0) < 0.15


def test_wide_frames_resized():
    done = {}
    rec = ClipRecorder(TMP / "e", max_width=1280, on_done=lambda eid, p: done.__setitem__(eid, p))
    feed(rec, 4, events=[(1, 2.0)], w=2560)
    rec.close()
    assert read_times(done[1])[1] == (1280, 180)


def test_slow_processing_keeps_real_speed():
    """Máy xử lý chỉ 8 FPS -> clip ghi 8 FPS để phát đúng tốc độ thật."""
    done = {}
    rec = ClipRecorder(TMP / "f", on_done=lambda eid, p: done.__setitem__(eid, p))
    feed(rec, 10, fps_in=8, events=[(1, 6.0)])
    rec.close()
    ts, _, fps = read_times(done[1])
    assert abs(fps - 8) < 0.6 and abs((len(ts) - 1) / fps - 5) < 0.3, (fps, len(ts))


def test_ring_buffer_bounded():
    rec = ClipRecorder(TMP / "g")
    feed(rec, 20)
    assert len(rec.buf) <= 5 * 15 + 1
    rec.close()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except AssertionError as e:
            failed += 1
            print(f"SAI  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test dat")
    sys.exit(1 if failed else 0)
