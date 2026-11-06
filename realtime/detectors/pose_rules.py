"""Luật tư thế: GIƠ TAY ĐẦU HÀNG, VẪY TAY CẦU CỨU, NẰM BẤT ĐỘNG, ÔM ĐẦU / GỤC ĐẦU (CLAUDE.md mục 9.2).
Mọi khoảng cách chuẩn hoá theo chiều dài thân (vai→hông; không thấy hông thì 1.2 × độ rộng vai)."""
from collections import deque

import numpy as np

from detectors.base import Detector

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
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


class PoseRulesDetector(Detector):
    name = "pose_rules"

    def __init__(self, fps=15, hands_up=True, hands_up_sec=2.5, wave=True, wave_sec=3.0, wave_turns=5,
                 wave_amplitude=0.4, lying_still=True, still_sec=10.0, still_speed=0.08,
                 distress=True, distress_sec=10.0, distress_slumped=True, cooldown_sec=15.0, **kw):
        super().__init__(fps, **kw)
        self.enabled = {HANDS_UP: hands_up, WAVE: wave, STILL: lying_still, DISTRESS: distress}
        self.need = {HANDS_UP: int(hands_up_sec * fps), STILL: int(still_sec * fps),
                     DISTRESS: int(distress_sec * fps)}
        self.wave_len, self.wave_turns, self.wave_amp = int(wave_sec * fps), wave_turns, wave_amplitude
        self.still_speed, self.slumped_on = still_speed, distress_slumped
        self.cooldown = int(cooldown_sec * fps)
        self.hist, self.count, self.cool = {}, {}, {}

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
        if len(q) < self.wave_len:
            return False
        arr = np.array(list(q)[-self.wave_len:])
        for wr, s in ((L_WR, L_SH), (R_WR, R_SH)):
            good = (arr[:, wr, 2] >= CONF) & (arr[:, s, 2] >= CONF)
            if good.mean() < 0.8:
                continue
            above = (arr[good, wr, 1] < arr[good, s, 1]).mean() > 0.8
            x = arr[good, wr, 0]
            if not above or (x.max() - x.min()) < self.wave_amp * torso:
                continue
            v = np.diff(x)
            v = v[np.abs(v) > 0.02 * torso]
            turns = np.sum(np.sign(v[1:]) != np.sign(v[:-1]))
            if turns >= self.wave_turns:
                return True
        return False

    def _lying_still(self, k, q, sh, torso, box):
        angle = torso_angle(k, sh)
        if angle is None or len(q) <= self.fps:
            return False
        w, h = box[2] - box[0], box[3] - box[1]
        if not (angle > 60 or w > 1.2 * h):
            return False
        a, b = q[-self.fps - 1], q[-1]                   # cách nhau 1 s
        m = (a[:, 2] >= CONF) & (b[:, 2] >= CONF)
        if m.sum() < 5:
            return False
        move = np.linalg.norm(b[m, :2] - a[m, :2], axis=1).mean() / torso
        return move < self.still_speed

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
            return []
        sh, torso = s
        up = self._hands_up(kpts, sh, torso)
        conds = {HANDS_UP: up,
                 STILL: self._lying_still(kpts, q, sh, torso, box),
                 DISTRESS: (not up) and self._distress(kpts, sh, torso)}
        rules = []
        for rule, c in conds.items():
            if not self.enabled[rule]:
                continue
            key = (tid, rule)
            self.count[key] = self.count.get(key, 0) + 1 if c else 0
            if self._fire(key, self.count[key] >= self.need[rule]):
                rules.append(rule)
        if self.enabled[WAVE] and self._fire((tid, WAVE), self._waving(q, torso)):
            rules.append(WAVE)
        return rules

    def update(self, frame, tracks, t):
        gone = set(self.gone_ids(tracks, t))
        if gone:
            for tid in gone:
                self.hist.pop(tid, None)
            self.count = {k: v for k, v in self.count.items() if k[0] not in gone}
            self.cool = {k: v for k, v in self.cool.items() if k[0] not in gone}
        return [self.event(rule, [tid]) for tid, tr in tracks.items()
                for rule in self._check(tid, tr["kpts"], tr["box"])]
