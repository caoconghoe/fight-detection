"""Unit test detectors/scene.py (luồng 2): logic cảnh báo và đếm cửa sổ liên tiếp, bằng bộ chấm điểm giả.
Chạy (từ thư mục httm):  python realtime/tests/test_scene_detector.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.events import PRIORITY  # noqa: E402
from detectors.scene import SceneDetector  # noqa: E402

GROUPS = ["BẠO LỰC", "CƯỚP", "CHÁY NỔ", "ĐỘT NHẬP/TRỘM", "TAI NẠN"]


class StubScorer:
    """Giả lập SceneScorer: mỗi lần drain() trả các cửa sổ đã định sẵn."""
    backend, groups = "stub", GROUPS

    def __init__(self, script):
        self.script = list(script)              # danh sách các điểm, mỗi phần tử = 1 cửa sổ
        self.submitted = 0

    def submit(self, t, frame):
        self.submitted += 1

    def drain(self):
        if not self.script:
            return []
        s, g = self.script.pop(0)
        return [(0.0, s, np.array(g, np.float32))]

    def close(self):
        pass


def run(scores, threshold=-0.8, min_consec=2, group=1):
    """Chạy detector với chuỗi điểm; mỗi cửa sổ ứng với 1 tick (2 FPS -> mỗi tick chỉ 1 frame mới)."""
    g = [0.0] * 5
    g[group] = 3.0
    det = SceneDetector(threshold=threshold, min_consecutive=min_consec, scorer=StubScorer([(s, g) for s in scores]))
    events = []
    for i in range(len(scores)):
        events += det.update(np.zeros((4, 4, 3), np.uint8), {}, i * 0.5)
    return events


def test_priority_is_50():
    assert PRIORITY["CẢNH BẤT THƯỜNG"] == 50


def test_two_consecutive_alert_once():
    ev = run([-2.0, -0.5, -0.4, -0.3, -0.2, -2.0])
    assert len(ev) == 1 and ev[0]["type"] == "CẢNH BẤT THƯỜNG" and ev[0]["ids"] == [], ev


def test_single_window_no_alert():
    assert run([-2.0, -0.5, -2.0, -0.5, -2.0]) == []


def test_group_is_hint_and_recorded():
    ev = run([-0.5, -0.4], group=1)
    assert ev[0]["scene_group"] == "CƯỚP" and "gợi ý" in ev[0]["detail"]
    assert ev[0]["severity"] == 50


def test_rearm_after_drop():
    ev = run([-0.5, -0.5, -2.0, -2.0, -0.5, -0.5])
    assert len(ev) == 2, ev


def test_no_alert_when_model_missing():
    det = SceneDetector(scorer=None, model_dir="/no/such/dir")
    assert det.update(np.zeros((4, 4, 3), np.uint8), {}, 0.0) == []


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
