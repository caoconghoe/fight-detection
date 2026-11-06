"""Unit test chống báo nhầm khi ngồi gần camera (TÉ NGÃ, GIƠ TAY) – chạy qua EventManager với
mọi detector bật trong config.yaml. Chạy (từ thư mục httm):  python realtime/tests/test_body_rules.py"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.events import EventManager, build_detectors  # noqa: E402
from synth import person, run  # noqa: E402

FRAME = (720, 1280)                                        # ảnh webcam 720p (H, W)
RNG = np.random.default_rng(0)


def manager():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    return EventManager(build_detectors(cfg), **cfg["events"])


def types(events):
    return [e[1] for e in events]


def ramp(t, start, dur):
    return float(np.clip((t - start) / dur, 0, 1))


def seated_close(t, hands_behind_neck, bow_start=2.0, bow_dur=0.6):
    """Ngồi sát webcam 720p: chỉ thấy nửa trên, khung chạm mép dưới, không thấy gối.
    Hông thật ở ngoài khung nhưng YOLO vẫn "đoán" (conf 0.6): lúc ngồi thẳng ở y=600, khi cúi trượt xuống 715.
    Từ bow_start: cúi người về phía camera + cúi đầu trong bow_dur giây: vai tụt 220 px, đoạn vai→hông
    co từ 170 px còn 65 px, mặt tụt xuống ngang vai.
    hands_behind_neck: đưa 2 tay lên đan sau gáy trong 0–1.5 s (khuỷu dạng rộng)."""
    b = ramp(t, bow_start, bow_dur)
    sh = np.array([640.0, 430 + 220 * b])
    hip = np.array([640.0, 600 + 115 * b])
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    nose = sh + [0, -130 + 130 * b]                         # cúi đầu: mặt tụt xuống ngang vai
    k[0, :2] = nose
    k[1, :2], k[2, :2] = nose + [20, -15], nose - [20, 15]
    k[3, :2], k[4, :2] = nose + [45, 0], nose - [45, 0]
    k[[0, 1, 2, 3, 4], 2] = 0.9 - 0.4 * b
    k[5, :2], k[6, :2] = sh + [110, 0], sh - [110, 0]
    k[11, :2], k[12, :2] = hip + [80, 0], hip - [80, 0]
    k[[11, 12], 2] = 0.6
    k[13:, 2] = 0.0                                         # gối, cổ chân ngoài khung
    if hands_behind_neck:
        up = ramp(t, 0.0, 1.5)
        for s, el, wr, sgn in ((5, 7, 9, 1), (6, 8, 10, -1)):
            down_el, down_wr = k[s, :2] + [sgn * 20, 150], k[s, :2] + [sgn * 40, 250]
            neck_el = k[s, :2] + [sgn * 90, -90]                          # khuỷu dạng ra, cao hơn vai
            neck_wr = sh + [sgn * 35, -143 + 40 * b]                      # cổ tay sau gáy; mặt cúi xuống thấp hơn tay
            k[el, :2] = down_el + (neck_el - down_el) * up
            k[wr, :2] = down_wr + (neck_wr - down_wr) * up
    else:
        for s, el, wr, sgn in ((5, 7, 9, 1), (6, 8, 10, -1)):
            k[el, :2], k[wr, :2] = k[s, :2] + [sgn * 20, 150], k[s, :2] + [sgn * 60, 230]
    k[:, :2] += RNG.normal(0, 2, (17, 2))
    k[k[:, 1] > FRAME[0], 2] = 0.0                          # ngoài ảnh -> không thấy
    vis = k[:, 2] > 0
    box = np.array([k[vis, 0].min() - 30, k[vis, 1].min() - 40, k[vis, 0].max() + 30, FRAME[0]], np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, FRAME[1])
    box[1] = max(box[1], 0)
    return {1: {"box": box, "kpts": k.astype(np.float32)}}


def test_hands_behind_neck_bow_no_alert():
    """Đan tay sau gáy rồi cúi đầu (8 s, < 10 s của ÔM ĐẦU) -> không báo gì."""
    ev = run(manager(), lambda t: seated_close(t, True), 8, frame_shape=FRAME)
    assert ev == [], ev


def test_seated_close_bow_no_fall():
    """Ngồi gần camera cúi người -> không TÉ NGÃ (và không báo gì)."""
    ev = run(manager(), lambda t: seated_close(t, False), 10, frame_shape=FRAME)
    assert ev == [], ev


def test_scene_is_wide_box_touching_bottom():
    """Kiểm tra kịch bản đúng như mô tả: khung rộng (w > 1.2h) và chạm mép dưới."""
    tr = seated_close(5.0, True)[1]
    x1, y1, x2, y2 = tr["box"]
    assert x2 - x1 > 1.2 * (y2 - y1) and y2 >= FRAME[0] - 1


def test_arms_straight_up_still_alerts():
    ev = run(manager(), lambda t: {1: person(640, 400, arms="up" if t >= 1 else "down", rng=RNG)}, 6,
             frame_shape=FRAME)
    assert types(ev) == ["GIƠ TAY ĐẦU HÀNG"], ev


def test_full_body_fall_still_alerts():
    def fall(t):
        x = ramp(t, 3, 0.6)
        return {1: person(640, 400 + 135 * x, a_deg=85 * x, rng=RNG)}
    ev = run(manager(), fall, 8, frame_shape=FRAME)
    assert types(ev) == ["TÉ NGÃ"], ev


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
