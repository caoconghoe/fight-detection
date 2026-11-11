"""CHẠY NHANH – luật khung xương, không huấn luyện.

Với mỗi ID (dữ liệu vào 15 FPS):
- Thang đo L = trung vị độ dài các đoạn conf ≥ seg_conf trong {vai–hông (2 bên), rộng vai, rộng hông, hông–gối (2 bên)},
  bỏ đoạn lệch > seg_max_dev so với trung vị rồi lấy trung vị lần nữa.
- Tốc độ NGANG (trong mặt phẳng ảnh) = dịch chuyển tâm hông mỗi giây / L.
- Tốc độ DỌC (lại gần / ra xa camera) = (f / L) × |hệ số góc của ln(L) theo thời gian trong 1 s|,
  f = chiều cao ảnh / (2·tan(vfov/2)). (Khoảng cách Z = f·L_thật/L_px ⇒ |dZ/dt| / L_thật = (f/L)·|d ln L/dt|.)
- Tốc độ TỔNG = √(ngang² + dọc²) từng frame, lấy trung vị trong 1 s.
  Chỉ tính tốc độ dọc khi L đổi đủ nhiều trong cửa sổ: |ln(L_cuối / L_đầu)| ≥ min_scale_change, với L_đầu / L_cuối
  lấy trên đường thẳng khớp ln(L) (không lấy 2 frame đơn lẻ – mỗi frame L rung ±10%); không đạt -> tốc độ dọc = 0.
Báo khi cùng lúc, liên tục ≥ min_sec: tổng > speed_thr; có động tác chân; hướng ổn định. Xem __init__ cho từng ngưỡng.
Không báo cho ID thuộc cặp có EMA đánh nhau ≥ fight_suppress, và trong fight_suppress_sec sau khi cặp hạ dưới mức đó.
"""
import math
from collections import deque

import numpy as np

from detectors.base import Detector

L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
SEGMENTS = [(5, 11), (6, 12), (5, 6), (11, 12), (11, 13), (12, 14)]
RUN = "CHẠY NHANH"
DIR_TEXT = {"ngang": "ngang", "lai_gan": "lại gần camera", "ra_xa": "ra xa camera"}


def body_scale(k, conf=0.5, max_dev=0.4):
    """L (pixel) từ keypoint (17, 3) hoặc None nếu không thấy đoạn nào."""
    lens = [float(np.hypot(*(k[a, :2] - k[b, :2]))) for a, b in SEGMENTS if k[a, 2] >= conf and k[b, 2] >= conf]
    lens = [v for v in lens if v > 1.0]
    if not lens:
        return None
    m = float(np.median(lens))
    keep = [v for v in lens if abs(v - m) <= max_dev * m]
    return float(np.median(keep)) if keep else m


def knee_angle(k, hip, knee, ank, conf):
    """Góc hông–gối–cổ chân (độ, 180 = duỗi thẳng) hoặc None."""
    if min(k[hip, 2], k[knee, 2], k[ank, 2]) < conf:
        return None
    a, b = k[hip, :2] - k[knee, :2], k[ank, :2] - k[knee, :2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def swings(x, amp):
    """Số lần tín hiệu đổi chiều với biên độ ≥ amp (có trễ): tăng ≥ amp rồi giảm ≥ amp = 2 lần.
    Thay đổi một chiều (đứng dậy, cúi người) chỉ cho ≤ 1 lần."""
    x = [v for v in x if v is not None]
    if len(x) < 3:
        return 0
    n, d, ext = 0, 0, x[0]                       # d: chiều hiện tại (+1 tăng, -1 giảm, 0 chưa rõ)
    lo = hi = x[0]
    for v in x[1:]:
        if d == 0:
            lo, hi = min(lo, v), max(hi, v)
            if hi - lo >= amp:
                d = 1 if v == hi else -1
                ext = v
                n = 1
        elif d > 0:
            if v > ext:
                ext = v
            elif ext - v >= amp:
                d, ext, n = -1, v, n + 1
        else:
            if v < ext:
                ext = v
            elif v - ext >= amp:
                d, ext, n = 1, v, n + 1
    return n


def beats(x, h):
    """Số lần đổi dấu (có trễ ±h) của chuỗi x – nhịp hai cổ chân lên xuống luân phiên."""
    n, s = 0, 0
    for v in x:
        if v is None:
            continue
        cur = 1 if v > h else (-1 if v < -h else 0)
        if cur != 0:
            if s != 0 and cur != s:
                n += 1
            s = cur
    return n


class RunningDetector(Detector):
    name = "running"

    def __init__(self, fps=15, speed_thr=4.5, min_sec=1.0, win_sec=1.0, camera_vfov_deg=50.0,
                 seg_conf=0.5, seg_max_dev=0.4, need_conf=0.5, ankle_conf=0.5,
                 knee_amp_deg=35.0, ankle_amp=0.6, leg_swings=2, alt_beats=2, alt_hyst=0.1,
                 straight_min=0.6, depth_sign_ratio=0.8, min_scale_change=0.15, fight_suppress=0.4, fight_suppress_sec=2.0,
                 jump_body=3.0, min_track_sec=1.0, shake_min_people=3, shake_ratio=0.7, shake_min_move=0.5,
                 cooldown_sec=10.0, default_height=720, **kw):
        super().__init__(fps, **kw)
        self.speed_thr, self.min_sec, self.win = speed_thr, min_sec, win_sec
        self.vfov = math.radians(camera_vfov_deg)
        self.seg_conf, self.seg_max_dev, self.need_conf, self.ankle_conf = seg_conf, seg_max_dev, need_conf, ankle_conf
        # động tác chân (một trong ba): góc gối hoặc khoảng cách 2 cổ chân dao động (nhìn ngang);
        # 2 cổ chân lên xuống luân phiên (nhìn chính diện)
        self.knee_amp, self.ankle_amp, self.leg_swings = knee_amp_deg, ankle_amp, leg_swings
        self.alt_beats, self.alt_hyst = alt_beats, alt_hyst
        # hướng ổn định: ngang = |dịch chuyển ròng| / quãng đường ≥ straight_min; dọc = dấu ln(L) giữ ≥ depth_sign_ratio
        self.straight_min, self.depth_sign_ratio = straight_min, depth_sign_ratio
        self.min_scale_change = min_scale_change
        # đang đánh nhau (EMA của detector fight) -> không báo chạy
        self.fight_suppress, self.fight_suppress_sec = fight_suppress, fight_suppress_sec
        self.fight, self.fight_t = None, {}                 # detector fight (gắn qua attach), id -> t lần cuối EMA ≥ mức
        # chặn báo nhầm
        self.jump_body, self.min_track = jump_body, min_track_sec
        self.shake_min_people, self.shake_ratio, self.shake_min_move = shake_min_people, shake_ratio, shake_min_move
        self.cooldown, self.default_h = cooldown_sec, default_height
        self.hist, self.first_t, self.run_start, self.run_dir = {}, {}, {}, {}
        self.fired, self.last_fire, self.dbg = set(), {}, {}
        self.shake = False

    # ---------- đo từng ID ----------
    def _sample(self, tid, tr, t, H):
        k, box = tr["kpts"], tr["box"]
        c = self.need_conf
        if min(k[L_HIP, 2], k[R_HIP, 2]) < c or max(k[L_KNEE, 2], k[R_KNEE, 2]) < c:
            return "thieu hong/goi"
        L = body_scale(k, self.seg_conf, self.seg_max_dev)
        if L is None:
            return "thieu doan than"
        hip = k[[L_HIP, R_HIP], :2].mean(0)
        q = self.hist.get(tid)
        if q:
            prev = q[-1]
            steps = max(1, round((t - prev["t"]) * self.fps))
            if np.hypot(*(hip - prev["hip"])) > self.jump_body * L * steps:   # ID nhảy -> làm lại như ID mới
                q.clear()
                self.first_t[tid] = t
                self._reset_run(tid)
        if q is None:
            q = self.hist[tid] = deque(maxlen=int(3 * self.win * self.fps) + 2)
            self.first_t[tid] = t
        margin = self.full_body["edge_margin"] * H
        legs_y = [k[i, 1] for i in (L_KNEE, R_KNEE, L_ANK, R_ANK) if k[i, 2] >= 0.3]
        edge = box[3] >= H - margin or any(y >= H - margin for y in legs_y)
        ank = None
        if min(k[L_ANK, 2], k[R_ANK, 2]) >= self.ankle_conf:
            ank = (k[L_ANK, :2].copy(), k[R_ANK, :2].copy())
        s = {"t": t, "hip": hip, "L": L, "lnL": math.log(L), "edge": edge, "ank": ank,
             "knee": (knee_angle(k, L_HIP, L_KNEE, L_ANK, self.ankle_conf),
                      knee_angle(k, R_HIP, R_KNEE, R_ANK, self.ankle_conf)),
             "vh": None, "vd": None, "slope": None}
        if q:
            prev = q[-1]
            dt = t - prev["t"]
            if dt > 1e-6:
                s["vh"] = float(np.hypot(*(hip - prev["hip"]))) / dt / (0.5 * (L + prev["L"]))
        q.append(s)
        win = [x for x in q if x["t"] >= t - self.win - 1e-6]
        if len(win) >= max(3, int(0.6 * self.win * self.fps)) and not any(x["edge"] for x in win):
            ts = np.array([x["t"] for x in win])
            slope = float(np.polyfit(ts - ts[0], [x["lnL"] for x in win], 1)[0])
            f = H / (2 * math.tan(self.vfov / 2))
            s["slope"] = slope
            s["dlnL"] = abs(slope) * float(ts[-1] - ts[0])  # |ln(L_cuối / L_đầu)| trên đường khớp
            s["vd"] = (f / float(np.median([x["L"] for x in win])) * abs(slope)
                       if s["dlnL"] >= self.min_scale_change else 0.0)
        return None

    def _measure(self, tid, t):
        """Số liệu 1 s gần nhất của ID, hoặc None nếu chưa đủ dữ liệu."""
        q = self.hist.get(tid)
        if not q:
            return None
        win = [x for x in q if x["t"] >= t - self.win - 1e-6]
        if len(win) < max(3, int(0.6 * self.win * self.fps)):
            return None
        Lm = float(np.median([x["L"] for x in win]))
        vh_list = [x["vh"] for x in win if x["vh"] is not None]
        vh = float(np.median(vh_list)) if vh_list else 0.0
        cur = win[-1]
        vd = cur["vd"]
        tot = [math.hypot(x["vh"], x["vd"] or 0.0) for x in win if x["vh"] is not None]
        vt = float(np.median(tot)) if tot else 0.0
        # động tác chân
        knee_sw = max(swings([x["knee"][j] for x in win], self.knee_amp) for j in (0, 1))
        ank_d = [float(np.hypot(*(x["ank"][0] - x["ank"][1]))) / x["L"] if x["ank"] else None for x in win]
        ank_sw = swings(ank_d, self.ankle_amp)
        alt = beats([(x["ank"][0][1] - x["ank"][1][1]) / x["L"] if x["ank"] else None for x in win], self.alt_hyst)
        legs = knee_sw >= self.leg_swings or ank_sw >= self.leg_swings or alt >= self.alt_beats
        # hướng
        net = win[-1]["hip"] - win[0]["hip"]
        path = sum(float(np.hypot(*(b["hip"] - a["hip"]))) for a, b in zip(win, win[1:]))
        straight = float(np.hypot(*net)) / path if path > 1e-6 else 0.0
        slopes = [x["slope"] for x in win if x["slope"] is not None]
        if vd is not None and vd > vh:
            mode = "lai_gan" if cur["slope"] > 0 else "ra_xa"
            sgn = 1 if cur["slope"] > 0 else -1
            stable = bool(slopes) and np.mean([np.sign(s) == sgn for s in slopes]) >= self.depth_sign_ratio
        else:
            mode, stable = "ngang", straight >= self.straight_min
        return dict(L=Lm, vh=vh, vd=vd, vt=vt, knee_sw=knee_sw, ank_sw=ank_sw, alt=alt, legs=legs,
                    mode=mode, stable=stable, straight=straight, net=net / Lm, edge=cur["edge"],
                    slope=cur["slope"], dlnL=cur.get("dlnL"))

    def _reset_run(self, tid):
        self.run_start.pop(tid, None)
        self.run_dir.pop(tid, None)
        self.fired.discard(tid)

    def _reversed(self, tid, m):
        """Đảo chiều so với lúc bắt đầu chuỗi chạy: ngang đổi chiều, hoặc ln(L) đổi dấu."""
        d0 = self.run_dir.get(tid)
        if d0 is None:
            return False
        net0, mode0 = d0
        n0, n1 = np.linalg.norm(net0), np.linalg.norm(m["net"])
        if n0 > 0.5 and n1 > 0.5 and np.dot(net0, m["net"]) < 0:
            return True
        return mode0 != "ngang" and m["mode"] != "ngang" and mode0 != m["mode"]

    def _camera_shake(self, meas):
        """≥ shake_ratio số người (trong ≥ shake_min_people người có số liệu) cùng dịch một hướng -> rung/lia camera."""
        if len(meas) < self.shake_min_people:
            return False
        dirs = [m["net"] / np.linalg.norm(m["net"]) for m in meas.values()
                if np.linalg.norm(m["net"]) >= self.shake_min_move]
        if len(dirs) < 2:
            return False
        mean = np.mean(dirs, 0)
        if np.linalg.norm(mean) < 1e-6:
            return False
        mean /= np.linalg.norm(mean)
        same = sum(float(np.dot(d, mean)) > 0.8 for d in dirs)
        return same / len(meas) >= self.shake_ratio

    # ---------- giao diện Detector ----------
    def attach(self, detectors):
        self.fight = next((d for d in detectors if d.name == "fight"), None)

    def _fighting(self, t):
        """Cập nhật thời điểm từng ID thuộc cặp có EMA đánh nhau ≥ fight_suppress; trả về tập ID đang bị chặn."""
        if self.fight is not None:
            for pair, e in getattr(self.fight, "ema", {}).items():
                if e >= self.fight_suppress:
                    for i in pair:
                        self.fight_t[i] = t
        self.fight_t = {i: v for i, v in self.fight_t.items() if t - v <= self.fight_suppress_sec + 1e-6}
        return set(self.fight_t)

    def update(self, frame, tracks, t):
        for tid in self.gone_ids(tracks, t):
            for d in (self.hist, self.first_t, self.dbg, self.last_fire):
                d.pop(tid, None)
            self._reset_run(tid)
        fighting = self._fighting(t)
        H = frame.shape[0] if frame is not None else self.default_h
        meas, why = {}, {}
        for tid, tr in tracks.items():
            r = self._sample(tid, tr, t, H)
            if r:
                why[tid] = r
                continue
            if t - self.first_t[tid] < self.min_track - 1e-6:
                why[tid] = "ID moi"
                continue
            m = self._measure(tid, t)
            if m is None:
                why[tid] = "chua du 1 s"
                continue
            meas[tid] = m
        self.shake = self._camera_shake(meas)

        events = []
        for tid in tracks:
            m = meas.get(tid)
            if m is None:
                self.dbg[tid] = {"why": why.get(tid, "?")}
                if why.get(tid) != "ID moi":            # mất khớp: chuỗi chạy đứt
                    self._reset_run(tid)
                continue
            m = dict(m)
            if m["edge"]:                               # chân chạm mép dưới: dừng đo dọc, giữ nguyên trạng thái
                m["state"] = "mep duoi (giu)"
                self.dbg[tid] = m
                continue
            if tid in fighting:                         # đang / vừa đánh nhau: không tính chạy, chuỗi chạy làm lại
                self._reset_run(tid)
                m["held"], m["state"] = 0.0, "danh nhau (chan)"
                self.dbg[tid] = m
                continue
            fast = m["vt"] > self.speed_thr
            cond = fast and m["legs"] and m["stable"] and not self.shake
            if cond and self._reversed(tid, m):
                self._reset_run(tid)
            if not cond:
                self._reset_run(tid)
            else:
                self.run_start.setdefault(tid, t)
                self.run_dir.setdefault(tid, (m["net"], m["mode"]))
                held = t - self.run_start[tid]
                if (held >= self.min_sec - 1e-6 and tid not in self.fired
                        and t - self.last_fire.get(tid, -1e9) >= self.cooldown):
                    self.fired.add(tid)
                    self.last_fire[tid] = t
                    events.append(self.event(RUN, [tid], detail=f"hướng: {DIR_TEXT[m['mode']]}, "
                                                                 f"tốc độ {m['vt']:.1f} thân/s"))
            m["held"] = t - self.run_start[tid] if tid in self.run_start else 0.0
            m["state"] = "rung camera" if self.shake else ""
            self.dbg[tid] = m
        return events

    def debug_info(self, tid):
        d = self.dbg.get(tid)
        if d is None:
            return []
        if "why" in d:
            return [f"CHAY: {d['why']}"]
        vd = "-" if d["vd"] is None else f"{d['vd']:.1f}"
        leg = f"goi{d['knee_sw']} cc{d['ank_sw']} lp{d['alt']}" + (" OK" if d["legs"] else "")
        return [f"CHAY ngang {d['vh']:.1f} doc {vd} tong {d['vt']:.1f}/{self.speed_thr}",
                f"  chan {leg} | {d['mode']}{' on dinh' if d['stable'] else ' DAO'} "
                f"| giu {d.get('held', 0):.1f}/{self.min_sec} {d.get('state', '')}"]
