"""Unit test luật tư thế (detectors/pose_rules.py) với ngưỡng thật trong config.yaml.
Chạy (từ thư mục httm):  python realtime/tests/test_pose_rules.py   (hoặc pytest nếu có cài)"""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectors.pose_rules import count_turns, PoseRulesDetector  # noqa: E402
from synth import FPS, person, run  # noqa: E402

WAVE = "VẪY TAY CẦU CỨU"
H = 300                                     # chiều cao người (px) -> thân (vai→hông) = 0.3 H = 90 px


def make(**override):
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    params = dict(cfg["detectors"]["pose_rules"])
    params.pop("enabled", None)
    params.update(override)
    return PoseRulesDetector(fps=cfg["sample_fps"], **params)


def types(events):
    return [e[1] for e in events]


def osc(t, hz):
    return math.sin(2 * math.pi * hz * t)


# ---- các kịch bản tay phải; toạ độ (ngang, lên) tính theo H, gốc là vai phải ----
# đầu (mũi) ở vai phải + (0.12, 0.15); đỉnh đầu ~ (0.12, 0.25)

def scratch_head(t, rng):
    """Vò đầu 1 tay: cổ tay trên đỉnh đầu, chà qua lại mạnh ±0.07 H (biên độ 0.47 thân) ở 2.5 Hz."""
    return {1: person(300, 400, H, rw=(0.12 + 0.07 * osc(t, 2.5), 0.25 + 0.02 * osc(t, 5)),
                      re=(-0.12, 0.12), rng=rng)}


def hand_on_head(t, rng):
    """Tay đặt trên đầu, cử động nhẹ ±0.02 H ở 2 Hz."""
    return {1: person(300, 400, H, rw=(0.12 + 0.02 * osc(t, 2), 0.24), re=(-0.12, 0.12), noise=2, rng=rng)}


def wave_above_shoulder(t, rng):
    """Vẫy tay bình thường: cổ tay cao hơn vai 0.2 H, lệch ra ngoài, vẫy ±0.1 H ở 1.5 Hz."""
    return {1: person(300, 400, H, rw=(-0.15 + 0.1 * osc(t, 1.5), 0.2), re=(-0.1, 0.08), rng=rng)}


def test_scratch_head_no_wave():
    assert WAVE not in types(run(make(), lambda t: scratch_head(t, RNG), 8))


def test_scratch_head_caught_by_near_head_filter():
    """Kiểm tra test có ý nghĩa: tắt bộ lọc 'gần đầu' thì vò đầu bị báo nhầm (đúng lỗi thấy khi thử thật)."""
    det = make(wave_near_head_ratio=1.01)
    assert WAVE in types(run(det, lambda t: scratch_head(t, RNG), 8))


def test_hand_on_head_no_wave():
    assert WAVE not in types(run(make(), lambda t: hand_on_head(t, RNG), 8))


def test_hand_on_head_small_moves_have_no_turns():
    """Cử động nhẹ (< 0.15 thân) không được đếm là đổi hướng, kể cả khi tắt bộ lọc gần đầu."""
    det = make(wave_near_head_ratio=1.01)
    assert WAVE not in types(run(det, lambda t: hand_on_head(t, RNG), 8))


def test_normal_wave_still_detected():
    ev = run(make(), lambda t: wave_above_shoulder(t, RNG), 8)
    assert types(ev).count(WAVE) == 1, ev                       # báo 1 lần (cooldown 15 s)
    assert ev[0][0] <= 3.5, ev                                  # báo ngay khi đủ 3 s


def test_hands_up_still_detected():
    ev = run(make(), lambda t: {1: person(300, 400, H, arms="up" if t >= 1 else "down", rng=RNG)}, 6)
    assert types(ev) == ["GIƠ TAY ĐẦU HÀNG"], ev


def test_count_turns_hysteresis():
    x = np.array([0, 10, 0, 10, 0, 10, 0], float)              # 5 lần đổi hướng, mỗi lần 10
    assert count_turns(x, 5) == 5
    assert count_turns(x, 15) == 0                              # mỗi lần dịch 10 < 15 -> không tính
    rng = np.random.default_rng(1)
    jitter = 30 * np.sin(np.arange(45) / 45 * 2 * np.pi) + rng.normal(0, 3, 45)   # 1 nhịp + rung nhỏ
    assert count_turns(jitter, 13.5) <= 2


RNG = np.random.default_rng(0)

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
