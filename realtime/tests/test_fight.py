"""Unit test detectors/fight.py (luật giả + nhánh ONNX với mô hình giả tests/fixtures/fake_pair_tcn.onnx).
Chạy (từ thư mục httm):  python realtime/tests/test_fight.py"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from detectors.fight import FightDetector  # noqa: E402
from shared import features as F  # noqa: E402
from synth import FPS, person, run  # noqa: E402

FAKE_ONNX = HERE / "fixtures" / "fake_pair_tcn.onnx"
TMP = Path(tempfile.mkdtemp())
RNG = np.random.default_rng(0)
FIGHT = "ĐÁNH NHAU"


def osc(t, hz):
    return math.sin(2 * math.pi * hz * t)


def scene(t, swing_a=False, swing_b=False, b=True, gap=200):
    """A ở x=500, B ở x=500+gap (H=300 -> 2 người cách 0.67 chiều cao). Vung tay phải 3 Hz về phía người kia."""
    out = {1: person(500, 450, H=300, rw=(-0.15 - 0.2 * osc(t, 3), 0.05) if swing_a else None, rng=RNG)}
    if b:
        out[2] = person(500 + gap, 450, H=300, rw=(0.15 + 0.2 * osc(t, 3 + 0.3), 0.05) if swing_b else None,
                        rng=RNG)
    return out


def fake_rule():
    return FightDetector(fps=FPS, model=TMP / "khong_co.onnx", model_config=TMP / "khong_co.json")


def onnx_det(**cfg_over):
    cfg = {"fps": 15, "window": 32, "stride": 8, "feat_dim": 180, "mean": [0.0] * 180, "std": [1.0] * 180,
           "threshold": 0.6, "pair_params": F.PARAMS, "pair_level": "test"}
    cfg.update(cfg_over)
    p = TMP / f"cfg_{len(list(TMP.iterdir()))}.json"
    p.write_text(json.dumps(cfg))
    return FightDetector(fps=FPS, model=FAKE_ONNX, model_config=p)


def types(ev):
    return [e[1] for e in ev]


# ---------------------------------------------------------------- luật giả
def test_fake_two_people_swinging_alert():
    det = fake_rule()
    assert det.mode == "luat gia"
    ev = run(det, lambda t: scene(t, True, True), 6)
    assert types(ev) == [FIGHT] and ev[0][2] == [1, 2], ev
    assert ev[0][0] <= 4.0, ev                                          # đủ bộ đệm + EMA + giữ 1 s


def test_fake_still_close_no_alert():
    assert run(fake_rule(), lambda t: scene(t), 8) == []


def test_fake_single_person_swinging_no_alert():
    assert run(fake_rule(), lambda t: scene(t, True, b=False), 8) == []


def test_fake_far_apart_swinging_no_alert():
    assert run(fake_rule(), lambda t: scene(t, True, True, gap=900), 8) == []


def test_highlight_and_reset():
    """Đang đánh: tô đỏ cặp; dừng đánh: hết tô đỏ; đánh lại: báo lại (EventManager lo cooldown 10 s)."""
    det = fake_rule()
    fired, hl_during, hl_after = [], set(), set()
    for i in range(int(14 * FPS)):
        t = i / FPS
        fighting = t < 5 or t >= 10
        fired += [(round(t, 1), e["ids"]) for e in det.update(None, scene(t, fighting, fighting), t)]
        if 4.5 < t < 5:
            hl_during |= det.highlight_ids()
        if 8 < t < 10:
            hl_after |= det.highlight_ids()
    assert hl_during == {1, 2} and hl_after == set(), (hl_during, hl_after)
    assert len(fired) == 2, fired


def test_id_gone_cleans_state():
    det = fake_rule()
    run(det, lambda t: scene(t, True, True) if t < 4 else {}, 7)
    assert det.ema == {} and det.buf == {} and det.highlight_ids() == set()


# ---------------------------------------------------------------- nhánh ONNX (mô hình giả)
def test_onnx_two_people_swinging_alert():
    det = onnx_det()
    assert det.mode == "onnx"
    ev = run(det, lambda t: scene(t, True, True), 6)
    assert types(ev) == [FIGHT] and ev[0][2] == [1, 2], ev


def test_onnx_still_and_single_no_alert():
    assert run(onnx_det(), lambda t: scene(t), 8) == []
    assert run(onnx_det(), lambda t: scene(t, True, b=False), 8) == []


def test_onnx_batch_many_pairs():
    """4 người đứng sát nhau -> nhiều cặp chạy chung 1 batch ONNX."""
    det = onnx_det()
    def crowd(t):
        return {i: person(400 + 120 * i, 450, H=300, rw=(-0.2 * osc(t, 3), 0.05), rng=RNG) for i in range(4)}
    ev = run(det, crowd, 6)
    assert len(det.ema) >= 3 and FIGHT in types(ev), (len(det.ema), ev)


def test_onnx_uses_pair_params_from_config():
    """pair_params trong fight_config.json được ưu tiên: near_ratio rất nhỏ -> không có cặp -> không báo."""
    det = onnx_det(pair_params={**F.PARAMS, "near_ratio": 0.1})
    assert det.pair["near_ratio"] == 0.1
    assert run(det, lambda t: scene(t, True, True), 6) == []


def test_onnx_threshold_from_config():
    det = onnx_det(threshold=0.99)
    assert det.threshold == 0.99


def test_bad_config_rejected():
    for bad in ({"feat_dim": 179}, {"fps": 30}, {"pair_params": {"min_presence": 0.5}}):
        try:
            onnx_det(**bad)
        except ValueError:
            continue
        raise AssertionError(f"phai bao loi: {bad}")


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
