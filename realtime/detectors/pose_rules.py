"""Luật tư thế: GIƠ TAY ĐẦU HÀNG, VẪY TAY CẦU CỨU, NẰM BẤT ĐỘNG, ÔM ĐẦU / GỤC ĐẦU (CLAUDE.md mục 9.2).
Mọi khoảng cách chuẩn hoá theo chiều dài thân (vai→hông; không thấy hông thì 1.2 × độ rộng vai)."""
from collections import deque

import numpy as np

from detectors.base import Detector

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
HEAD_KPTS = [0, 1, 2, 3, 4]
CONF = 0.4
HANDS_UP, WAVE, STILL, DISTRESS = "GIƠ TAY ĐẦU HÀNG", "VẪY TAY CẦU CỨU", "NẰM BẤT ĐỘNG", "ÔM ĐẦU / GỤC ĐẦU"


def ok(k, *idx):
    return all(k[i, 2] >= CONF for i in idx)


def torso_angle(k, sh):
    """Góc thân so với phương thẳng đứng (độ), None nếu không thấy hông."""
    if not ok(k, L_HIP, R_HIP):
        return None
    dx, dy = k[[L_HIP, R_HIP], :2].mean(0) - sh
    return np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))


def count_turns(x, min_swing):
    """Đếm số lần đổi hướng của chuỗi x; chỉ tính khi đã lùi lại ≥ min_swing từ điểm cực trị
    (hysteresis) -> rung keypoint nhỏ không được đếm."""
    if len(x) < 2:
        return 0
    lo = hi = ext = x[0]
    direction, turns = 0, 0
    for v in x[1:]:
        if direction == 0:
            lo, hi = min(lo, v), max(hi, v)
            if v - lo >= min_swing:
                direction, ext = 1, v
            elif hi - v >= min_swing:
                direction, ext = -1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v >= min_swing:
                turns, direction, ext = turns + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext >= min_swing:
                turns, direction, ext = turns + 1, 1, v
    return turns


class PoseRulesDetector(Detector):
    name = "pose_rules"

    def __init__(self, fps=15, hands_up=True, hands_up_sec=2.5,
                 wave=True, wave_sec=3.0, wave_turns=5, wave_amplitude=0.4, wave_min_swing=0.15,
                 wave_head_dist=0.5, wave_near_head_ratio=0.6,
                 lying_still=True, still_sec=10.0, still_speed=0.08,
                 distress=True, distress_sec=10.0, distress_slumped=True, cooldown_sec=15.0, **kw):
        super().__init__(fps, **kw)
        self.enabled = {HANDS_UP: hands_up, WAVE: wave, STILL: lying_still, DISTRESS: distress}
        self.need = {HANDS_UP: int(hands_up_sec * fps), STILL: int(still_sec * fps),
                     DISTRESS: int(distress_sec * fps)}
        self.wave_len, self.wave_turns, self.wave_amp = int(wave_sec * fps), wave_turns, wave_amplitude
        self.wave_swing, self.wave_head_dist, self.wave_near = wave_min_swing, wave_head_dist, wave_near_head_ratio
        self.still_speed, self.slumped_on = still_speed, distress_slumped
        self.cooldown = int(cooldown_sec * fps)
        self.hist, self.count, self.cool, self.dbg = {}, {}, {}, {}

    def _scale(self, k):
        if not ok(k, L_SH, R_SH):
            return None
        sh = k[[L_SH, R_SH], :2].mean(0)
        if ok(k, L_HIP, R_HIP):
            torso = np.linalg.norm(k[[L_HIP, R_HIP], :2].mean(0) - sh)
        else:
            torso = 1.2 * np.linalg.norm(k[L_SH, :2] - k[R_SH, :2])
        return (sh, torso) if torso > 1 else None

    def _head(self, k):
        vis = [i for i in HEAD_KPTS if k[i, 2] >= CONF]
        return k[vis, :2].mean(0) if vis else None

    def _hands_up(self, k, sh, torso):
        if not ok(k, L_WR, R_WR):
            return False
        head = self._head(k)
        top = head[1] if head is not None else sh[1] - 0.5 * torso
        return k[L_WR, 1] < top - 0.1 * torso and k[R_WR, 1] < top - 0.1 * torso

    def _waving(self, q, torso):
        """Vẫy tay (giữ điều kiện mục 9.2) = trong wave_sec giây: cổ tay cao hơn vai ở > 80% frame,
        biên độ ngang ≥ wave_amplitude thân, ≥ wave_turns lần đổi hướng – mỗi lần dịch ≥ wave_min_swing thân
        (bỏ rung tay nhỏ khi vò đầu). Loại trừ: cổ tay cách đầu < wave_head_dist thân ở
        ≥ wave_near_head_ratio số frame (tay đang ở trên đầu: vò/gãi/đặt tay lên đầu).
        Trả về (có vẫy?, thông tin của tay "gần đạt" nhất để debug)."""
        if len(q) < self.wave_len:
            return False, {"frames": f"{len(q)}/{self.wave_len}"}
        arr = np.array(list(q)[-self.wave_len:])
        hv = arr[:, HEAD_KPTS, 2] >= CONF                               # (N, 5) khớp đầu nhìn thấy
        has_head = hv.any(1)
        head = (arr[:, HEAD_KPTS, :2] * hv[..., None]).sum(1) / np.maximum(hv.sum(1), 1)[:, None]
        best = None
        for wr, s in ((L_WR, L_SH), (R_WR, R_SH)):
            good = (arr[:, wr, 2] >= CONF) & (arr[:, s, 2] >= CONF)
            info = {"turns": 0, "amp": 0.0, "above": 0.0, "near": 0.0, "good": good.mean()}
            if good.sum() >= 2:
                a = arr[good]
                info["above"] = (a[:, wr, 1] < a[:, s, 1]).mean()              # cổ tay cao hơn vai
                dist = np.linalg.norm(a[:, wr, :2] - head[good], axis=1)
                info["near"] = (has_head[good] & (dist < self.wave_head_dist * torso)).mean()
                x = a[:, wr, 0]
                info["amp"] = (x.max() - x.min()) / torso
                info["turns"] = count_turns(x, self.wave_swing * torso)
            info["ok"] = (info["good"] >= 0.8 and info["above"] > 0.8 and info["amp"] >= self.wave_amp
                          and info["turns"] >= self.wave_turns and info["near"] < self.wave_near)
            if best is None or (info["ok"], info["turns"], info["amp"]) > (best["ok"], best["turns"], best["amp"]):
                best = info
        return best["ok"], best

    def _lying_still(self, k, q, sh, torso, box):
        angle = torso_angle(k, sh)
        if angle is None or len(q) <= self.fps:
            return False, None
        a, b = q[-self.fps - 1], q[-1]                   # cách nhau 1 s
        m = (a[:, 2] >= CONF) & (b[:, 2] >= CONF)
        move = np.linalg.norm(b[m, :2] - a[m, :2], axis=1).mean() / torso if m.sum() >= 5 else None
        w, h = box[2] - box[0], box[3] - box[1]
        lying = angle > 60 or w > 1.2 * h
        return lying and move is not None and move < self.still_speed, move

    def _distress(self, k, sh, torso):
        head = self._head(k)
        if head is None:
            return False
        holding = ok(k, L_WR, R_WR) and max(
            np.linalg.norm(k[L_WR, :2] - head),
            np.linalg.norm(k[R_WR, :2] - head)) < 0.5 * torso
        angle = torso_angle(k, sh)
        upright = angle is None or angle < 45                # người nằm: đầu thấp hơn vai là bình thường
        slumped = self.slumped_on and upright and head[1] > sh[1] + 0.1 * torso
        return holding or slumped

    def _fire(self, key, cond):
        if self.cool.get(key, 0) > 0:
            self.cool[key] -= 1
            return False
        if cond:
            self.cool[key] = self.cooldown
            return True
        return False

    def _check(self, tid, kpts, box):
        q = self.hist.setdefault(tid, deque(maxlen=max(self.wave_len, self.fps) + 1))
        q.append(kpts.copy())
        s = self._scale(kpts)
        if s is None:
            self.dbg[tid] = {"torso": None}
            return []
        sh, torso = s
        up = self._hands_up(kpts, sh, torso)
        still, move = self._lying_still(kpts, q, sh, torso, box)
        conds = {HANDS_UP: up, STILL: still,
                 DISTRESS: (not up) and self._distress(kpts, sh, torso)}
        rules = []
        for rule, c in conds.items():
            if not self.enabled[rule]:
                continue
            key = (tid, rule)
            self.count[key] = self.count.get(key, 0) + 1 if c else 0
            if self._fire(key, self.count[key] >= self.need[rule]):
                rules.append(rule)
        wave, winfo = self._waving(q, torso) if self.enabled[WAVE] else (False, {})
        if self.enabled[WAVE] and self._fire((tid, WAVE), wave):
            rules.append(WAVE)
        self.dbg[tid] = {"torso": torso, "move": move, "wave": winfo}
        return rules

    def update(self, frame, tracks, t):
        gone = set(self.gone_ids(tracks, t))
        if gone:
            for tid in gone:
                self.hist.pop(tid, None)
                self.dbg.pop(tid, None)
            self.count = {k: v for k, v in self.count.items() if k[0] not in gone}
            self.cool = {k: v for k, v in self.cool.items() if k[0] not in gone}
        return [self.event(rule, [tid]) for tid, tr in tracks.items()
                for rule in self._check(tid, tr["kpts"], tr["box"])]

    def debug_info(self, tid):
        d = self.dbg.get(tid)
        if d is None:
            return []
        if d["torso"] is None:
            return ["tu the: khong thay 2 vai"]
        c = lambda r: f"{self.count.get((tid, r), 0)}/{self.need[r]}"
        cd = lambda r: self.cool.get((tid, r), 0) / self.fps
        lines = [f"than {d['torso']:.0f}px"]
        for label, r in (("GIO TAY", HANDS_UP), ("OM DAU", DISTRESS)):
            if self.enabled[r]:
                lines.append(f"{label} {c(r)}" + (f" (cho {cd(r):.0f}s)" if cd(r) > 0 else ""))
        if self.enabled[STILL]:
            mv = "-" if d["move"] is None else f"{d['move']:.2f}"
            lines.append(f"NAM IM {c(STILL)} v={mv}/{self.still_speed}")
        w = d["wave"]
        if self.enabled[WAVE]:
            if "turns" in w:
                lines.append(f"VAY doi huong {w['turns']}/{self.wave_turns} bien {w['amp']:.2f}/{self.wave_amp}"
                             + (f" (cho {cd(WAVE):.0f}s)" if cd(WAVE) > 0 else ""))
                lines.append(f"  tren vai {w['above']:.0%}/>80% gan dau {w['near']:.0%}/<{self.wave_near:.0%}")
            else:
                lines.append(f"VAY: dang gom {w.get('frames', '')} frame")
        return lines
