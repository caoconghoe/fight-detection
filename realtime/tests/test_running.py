"""Unit test detectors/running.py (CHẠY NHANH) bằng khung xương tổng hợp.
Chạy (từ thư mục httm):  python realtime/tests/test_running.py"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.events import PRIORITY, EventManager  # noqa: E402
from detectors.base import Detector  # noqa: E402
from detectors.running import RUN, RunningDetector, body_scale, swings  # noqa: E402

FPS = 15
FRAME = (1080, 1920)                                   # (H, W)
F = FRAME[0] / (2 * math.tan(math.radians(50) / 2))   # tiêu cự pixel khớp camera_vfov_deg = 50


def body(hip, L, view="side", t=0.0, cadence=0.0, thigh_deg=0.0, knee_deg=0.0, lift=0.0, sit=0.0,
         bend_deg=0.0, knee_flex_both=0.0, noise=0.5, rng=None):
    """Một người, đơn vị độ dài L (≈ thang đo body_scale): thân 1.2 L, rộng vai 1.0 L, rộng hông 0.7 L, đùi = cẳng = 1.0 L.
    view "side": chân đánh ra trước/sau (góc đùi ±thigh_deg, gối gập 0..knee_deg) – 2 chân lệch pha 180°.
    view "front": 2 cổ chân luân phiên nhấc lên tối đa lift × L.
    sit: 0 đứng .. 1 ngồi (nhìn chính diện: đùi co ngắn còn 0.2 L). bend_deg: cúi thân về trước (nhìn ngang).
    knee_flex_both: gập 2 gối CÙNG lúc (độ) – nhảy tại chỗ."""
    rng = rng or np.random.default_rng(0)
    hip = np.asarray(hip, float)
    k = np.zeros((17, 2))
    side = view == "side"
    hw, sw = (0.1, 0.15) if side else (0.35, 0.5)
    b = math.radians(bend_deg)
    up = np.array([math.sin(b), -math.cos(b)])         # hông -> vai
    sh = hip + 1.2 * L * up
    k[0] = sh + 0.45 * L * up + ([0.15 * L, 0] if side else [0, 0])
    k[1], k[2] = k[0] + [0.1 * L, -0.05 * L], k[0] + [-0.1 * L, -0.05 * L]
    k[3], k[4] = k[0] + [0.2 * L, 0], k[0] - [0.2 * L, 0]
    k[5], k[6] = sh + [sw * L, 0], sh - [sw * L, 0]
    k[7], k[8] = k[5] + [0, 0.7 * L], k[6] + [0, 0.7 * L]
    k[9], k[10] = k[7] + [0, 0.6 * L], k[8] + [0, 0.6 * L]
    k[11], k[12] = hip + [hw * L, 0], hip - [hw * L, 0]
    thigh = (1.0 - 0.8 * sit) * L
    for j, (hi, kn, an) in enumerate(((11, 13, 15), (12, 14, 16))):
        ph = 2 * math.pi * cadence * t + j * math.pi
        if side:
            th = math.radians(thigh_deg) * math.sin(ph)
            fl = math.radians(knee_deg) * (0.5 + 0.5 * math.sin(ph + math.pi / 2)) + math.radians(knee_flex_both)
            k[kn] = k[hi] + thigh * np.array([math.sin(th), math.cos(th)])
            k[an] = k[kn] + L * np.array([math.sin(th - fl), math.cos(th - fl)])
        else:
            k[kn] = k[hi] + [0, thigh]
            k[an] = k[kn] + [0, L - lift * L * max(0.0, math.sin(ph))]
    k += rng.normal(0, noise, k.shape)
    kp = np.hstack([k, np.full((17, 1), 0.9)]).astype(np.float32)
    m = 0.1 * L
    box = np.array([k[:, 0].min() - m, k[:, 1].min() - m, k[:, 0].max() + m, k[:, 1].max() + m], np.float32)
    return {"box": box, "kpts": kp}


def run(scene, secs, det=None):
    """scene(t, rng) -> tracks. Trả về (sự kiện [(t, ids, detail)], detector, tốc độ tổng lớn nhất từng ID)."""
    det = det or RunningDetector(fps=FPS)
    frame = np.zeros((*FRAME, 3), np.uint8)
    rng = np.random.default_rng(1)
    out, vmax = [], {}
    for i in range(int(secs * FPS)):
        t = i / FPS
        for e in det.update(frame, scene(t, rng), t):
            out.append((round(t, 2), e["ids"], e["detail"]))
        for tid, d in det.dbg.items():
            if "vt" in d:
                vmax[tid] = max(vmax.get(tid, 0.0), d["vt"])
    return out, det, vmax


# ---------- các cảnh ----------
def side_mover(speed_L, L=50, cadence=1.4, thigh=35, knee=90, x0=200, y=500):
    return lambda t, rng: {1: body((x0 + speed_L * L * t, y), L, "side", t, cadence, thigh, knee, rng=rng)}


def depth_mover(speed_L, z0, cadence=1.4, lift=0.6):
    """Chạy/đi thẳng theo trục camera: Z (đơn vị L thật) đổi speed_L mỗi giây; L_px = F / Z (dương = lại gần)."""
    def f(t, rng):
        L = F / (z0 - speed_L * t)
        return {1: body((960, 400 + 0.5 * L), L, "front", t, cadence, lift=lift, rng=rng)}
    return f


def test_body_scale_rejects_outliers():
    tr = body((500, 500), 50, "side", noise=0)
    assert abs(body_scale(tr["kpts"]) - 55) < 3, body_scale(tr["kpts"])      # bỏ rộng vai/hông bị co khi nhìn ngang
    tr = body((500, 500), 50, "front", noise=0)
    assert abs(body_scale(tr["kpts"]) - 50) < 3


def test_swings_monotonic_vs_oscillating():
    assert swings(list(np.linspace(90, 180, 15)), 35) == 1
    assert swings([140 + 50 * math.sin(2 * math.pi * 1.4 * i / 15) for i in range(16)], 35) >= 2


def test_run_sideways():
    ev, _, vmax = run(side_mover(8.0), 3.5)
    assert len(ev) == 1, ev
    assert "ngang" in ev[0][2] and ev[0][1] == [1], ev
    assert 1.8 <= ev[0][0] <= 2.6, ev                   # ID mới 1 s + cửa sổ 1 s + giữ 1 s (đếm từ lúc đủ cửa sổ)


def test_run_toward_camera():
    ev, _, vmax = run(depth_mover(8.0, z0=40), 3.0)
    assert len(ev) == 1, (ev, vmax)
    assert "lại gần camera" in ev[0][2], ev


def test_run_away_from_camera():
    ev, _, vmax = run(depth_mover(-8.0, z0=16), 3.0)
    assert len(ev) == 1, (ev, vmax)
    assert "ra xa camera" in ev[0][2], ev


def test_walk_sideways_no_alert():
    ev, _, vmax = run(side_mover(3.0, cadence=0.9, thigh=20, knee=40), 4.0)
    assert ev == [], (ev, vmax)


def test_walk_toward_camera_no_alert():
    ev, _, vmax = run(depth_mover(3.0, z0=40, cadence=0.9, lift=0.3), 4.0)
    assert ev == [], (ev, vmax)


def test_stand_up_from_chair_no_alert():
    def scene(t, rng):
        s = 1.0 - float(np.clip((t - 2.0) / 0.7, 0, 1))      # ngồi -> đứng trong 0.7 s
        L = 60
        return {1: body((960, 600 - 0.8 * L * (1 - s)), L, "front", sit=s, rng=rng)}
    ev, _, vmax = run(scene, 4.0)
    assert ev == [], (ev, vmax)


def test_bend_over_no_alert():
    def scene(t, rng):
        a = 70 * float(np.clip((t - 1.5) / 0.6, 0, 1) - np.clip((t - 3.0) / 0.6, 0, 1))
        return {1: body((960, 600), 60, "side", bend_deg=a, rng=rng)}
    ev, _, vmax = run(scene, 4.5)
    assert ev == [], (ev, vmax)


def test_jump_in_place_no_alert():
    def scene(t, rng):
        ph = 2 * math.pi * 2.2 * t
        L = 50
        return {1: body((960, 600 - 0.9 * L * abs(math.sin(ph))), L, "side",
                        knee_flex_both=50 * (1 - abs(math.sin(ph))), rng=rng)}
    ev, det, vmax = run(scene, 4.0)
    assert vmax.get(1, 0) > det.speed_thr, vmax       # đủ nhanh: chặn là nhờ điều kiện hướng ổn định
    assert ev == [], (ev, vmax)


def test_id_switch_once_no_alert():
    """Đi bộ ngang; tại 2.5 s ID 1 nhảy sang một người khác to gấp 1.5 lần, cách 6 L (ByteTrack đổi ID)."""
    walk = side_mover(3.0, cadence=0.9, thigh=20, knee=40)
    def scene(t, rng):
        if t < 2.5:
            return walk(t, rng)
        L = 75
        return {1: body((200 + 3 * 50 * t + 300, 520), L, "side", t, 0.9, 20, 40, rng=rng)}
    ev, det, vmax = run(scene, 5.0)
    assert ev == [], (ev, vmax)
    _, det, _ = run(scene, 2.6)                         # ngay sau cú nhảy: ID bị đo lại từ đầu như ID mới
    assert det.dbg[1].get("why") == "ID moi", det.dbg[1]


def test_camera_pan_no_alert():
    """3 người bước tại chỗ, camera lia ngang: cả 3 cùng dịch 8 L/s một hướng -> coi là rung/lia camera."""
    def scene(t, rng):
        return {i: body((200 + 300 * i + 8 * 50 * t, 500), 50, "side", t, 1.4, 35, 90, rng=rng) for i in (1, 2, 3)}
    ev, det, vmax = run(scene, 4.0)
    assert ev == [], (ev, vmax)


def test_one_runner_among_standing_people():
    def scene(t, rng):
        tr = {i: body((900 + 250 * i, 500), 50, "side", rng=rng) for i in (2, 3)}
        tr[1] = body((100 + 8 * 50 * t, 700), 50, "side", t, 1.4, 35, 90, rng=rng)
        return tr
    ev, _, _ = run(scene, 3.5)
    assert [e[1] for e in ev] == [[1]], ev


def test_no_vertical_speed_at_bottom_edge_and_alert_kept():
    """Chạy lại gần tới khi chân chạm mép dưới: sau đó không đo tốc độ dọc, cảnh báo đã có không bị báo lại."""
    def scene(t, rng):
        L = F / (40 - 8.0 * t)
        return {1: body((960, 300 + 6 * L), L, "front", t, 1.4, lift=0.6, rng=rng)}   # chân chạm mép ~3.4 s
    ev, det, _ = run(scene, 3.9)
    assert len(ev) == 1 and ev[0][0] < 3.3, ev
    assert det.dbg[1].get("vd") is None or det.dbg[1].get("state", "").startswith("mep"), det.dbg[1]
    assert 1 in det.fired


def test_standing_jittery_scale_has_no_vertical_speed():
    """Đứng yên, L mỗi frame dao động ngẫu nhiên ±10% (như YOLO thật) -> tốc độ dọc = 0 (min_scale_change)."""
    def scene(t, rng):
        tr = body((960, 600), 60, "front", rng=rng)
        k = tr["kpts"]
        hip = k[[11, 12], :2].mean(0)
        k[:, :2] = hip + (k[:, :2] - hip) * rng.uniform(0.9, 1.1)
        tr["box"] = np.array([k[:, 0].min() - 6, k[:, 1].min() - 6, k[:, 0].max() + 6, k[:, 1].max() + 6], np.float32)
        return {1: tr}
    # Nhiễu trắng ±10% mỗi frame: đường khớp 1 s còn lệch σ ≈ 0.047 nên ~0.1% cửa sổ vẫn vượt 0.15 (lẻ tẻ 1–2 tick);
    # tốc độ tổng là trung vị 1 s nên không bị ảnh hưởng. Kiểm tra: vd = 0 ở ≥ 99.5% cửa sổ, tổng luôn < 1 thân/s.
    vds, raw, vts = [], [], []
    for seed in range(20):
        det = RunningDetector(fps=FPS)
        frame = np.zeros((*FRAME, 3), np.uint8)
        rng = np.random.default_rng(seed)
        for i in range(int(6 * FPS)):
            t = i / FPS
            assert det.update(frame, scene(t, rng), t) == []
            d = det.dbg.get(1, {})
            if d.get("vd") is not None:
                vds.append(d["vd"])
                vts.append(d["vt"])
                raw.append(F / d["L"] * abs(d["slope"]))
    vds = np.array(vds)
    assert np.mean(vds > 0) <= 0.005, np.mean(vds > 0)
    assert max(vts) < 1.0, max(vts)
    assert np.percentile(raw, 95) > 1.5, np.percentile(raw, 95)   # không có ngưỡng thì nhiễu đã cho tốc độ dọc đáng kể


class FightStub(Detector):
    """Giả lập detector fight: ema[(a, b)] = ema_fn(t), cập nhật trước running mỗi tick (như thứ tự trong config)."""
    name = "fight"

    def __init__(self, ema_fn):
        super().__init__(FPS)
        self.ema_fn, self.ema = ema_fn, {}

    def update(self, frame, tracks, t):
        self.ema = {(1, 2): self.ema_fn(t)}
        return []


def run_with_fight(scene, ema_fn, secs):
    em = EventManager([FightStub(ema_fn), RunningDetector(fps=FPS)], panic_hold_sec=10.0)
    frame = np.zeros((*FRAME, 3), np.uint8)
    rng = np.random.default_rng(1)
    out = []
    for i in range(int(secs * FPS)):
        t = i / FPS
        out += [(round(t, 2), e["type"], e["ids"]) for e in em.update(frame, scene(t, rng), t)[0]]
    return out


def test_no_run_alert_while_fighting():
    """ID 1 di chuyển nhanh, có động tác chân (vờn / đuổi đánh) nhưng cặp (1, 2) có EMA đánh nhau 0.8 -> không báo."""
    def scene(t, rng):
        return {1: body((300 + 8 * 40 * t, 600), 40, "side", t, 1.4, 35, 90, rng=rng),
                2: body((340 + 8 * 40 * t, 600), 40, "side", t, 1.4, 35, 90, rng=rng)}
    assert run_with_fight(scene, lambda t: 0.8, 4.5) == []
    assert len(run_with_fight(scene, lambda t: 0.1, 4.5)) == 2       # cùng cảnh, không đánh nhau -> có báo


def test_run_away_after_fight_alerts():
    """Đánh nhau tại chỗ tới 3 s (EMA 0.8), sau đó EMA hạ 0.2 và ID 1 chạy đi -> báo, nhưng không sớm hơn 3 + 2 s."""
    def scene(t, rng):
        x1 = 300 + 8 * 40 * max(0.0, t - 3.0)
        cad = 1.4 if t >= 3.0 else 0.0
        return {1: body((x1, 600), 40, "side", t, cad, 35, 90, rng=rng),
                2: body((260, 600), 40, "side", rng=rng)}
    ev = run_with_fight(scene, lambda t: 0.8 if t < 3.0 else 0.2, 7.5)
    assert [(e[1], e[2]) for e in ev] == [(RUN, [1])], ev
    assert ev[0][0] >= 5.0, ev


class Stub(Detector):
    def __init__(self, script):
        super().__init__(FPS)
        self.script = script

    def update(self, frame, tracks, t):
        return [self.event(typ, ids) for tt, typ, ids in self.script if abs(tt - t) < 1e-6]


def test_event_manager_priority_and_panic_suppression():
    assert PRIORITY[RUN] == 55
    em = EventManager([Stub([(0.0, RUN, [1]), (1.0, "HOẢNG LOẠN", [2, 3, 4]), (5.0, RUN, [5]), (12.0, RUN, [6])])],
                      panic_hold_sec=10.0)
    got = []
    for t in (0.0, 1.0, 5.0, 12.0):
        got += [(t, e["type"]) for e in em.update(None, {}, t)[0]]
    assert got == [(0.0, RUN), (1.0, "HOẢNG LOẠN"), (12.0, RUN)], got


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
