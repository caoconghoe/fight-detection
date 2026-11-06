"""ĐÁNH NHAU – Pair-TCN (ONNX) trên cặp người, CLAUDE.md mục 7.4.

- Mỗi track giữ bộ đệm `window` frame mẫu gần nhất (15 FPS). Mỗi `infer_every` tick: dựng cặp ứng viên bằng
  CÙNG hàm với lúc train (shared/features.py, tiêu chí `pair_params` trong fight_config.json), gom 1 batch,
  chạy ONNX Runtime (CPU), sigmoid.
- Mỗi cặp làm mượt bằng EMA (α); EMA > threshold liên tục ≥ hold_sec -> báo ĐÁNH NHAU; highlight_ids = cặp đó.
- Chưa có pair_tcn.onnx / fight_config.json -> LUẬT GIẢ: 2 người khoảng cách tâm TB < fake_dist × chiều cao
  và tốc độ TB của cổ tay nhanh nhất (của người nhanh hơn) > fake_wrist_speed chiều cao/giây trong 1 s gần nhất.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np

from detectors.base import Detector
from shared import features as F

ROOT = Path(__file__).resolve().parents[1]                     # thư mục realtime/
WRISTS = [9, 10]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FightDetector(Detector):
    name = "fight"

    def __init__(self, fps=15, model="models/pair_tcn.onnx", model_config="models/fight_config.json",
                 infer_every=4, ema_alpha=0.4, threshold=0.6, hold_sec=1.0,
                 fake_dist=1.0, fake_wrist_speed=2.0, **kw):
        super().__init__(fps, **kw)
        self.infer_every, self.alpha, self.hold = infer_every, ema_alpha, hold_sec
        self.fake_dist, self.fake_speed = fake_dist, fake_wrist_speed
        self.threshold = threshold
        self.pair = dict(F.PARAMS)
        self.session = None
        model, model_config = self._path(model), self._path(model_config)
        if model.is_file() and model_config.is_file():
            self._load(model, model_config)
            self.mode = "onnx"
        else:
            self.mode = "luat gia"
            print(f"[fight] chua co {model.name} / {model_config.name} -> dung LUAT GIA (muc 7.4)")
        self.window = self.pair["window"]
        self.buf = {}                       # tid -> deque[(tick, box, kpts)]
        self.tick = 0
        self.ema = {}                       # (a, b) -> EMA xác suất
        self.last_p = {}                    # (a, b) -> xác suất lần chạy gần nhất (debug)
        self.above = {}                     # (a, b) -> t bắt đầu EMA > ngưỡng
        self.fired = set()                  # cặp đã báo trong đợt EMA > ngưỡng hiện tại
        self.fake_dbg = {}                  # (a, b) -> (khoảng cách, tốc độ cổ tay) cho debug

    @staticmethod
    def _path(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    def _load(self, model, model_config):
        cfg = json.loads(model_config.read_text(encoding="utf-8"))
        for key, want in (("fps", self.fps), ("feat_dim", F.FEAT_DIM)):
            if cfg.get(key) != want:
                raise ValueError(f"{model_config.name}: {key} = {cfg.get(key)} nhung realtime can {want}")
        unknown = set(cfg.get("pair_params", {})) - set(F.PARAMS)
        if unknown:
            raise ValueError(f"{model_config.name}: pair_params co khoa la {sorted(unknown)}")
        self.pair = {**F.PARAMS, **cfg.get("pair_params", {})}     # ưu tiên tiêu chí lúc train
        if cfg.get("window", self.pair["window"]) != self.pair["window"]:
            raise ValueError(f"{model_config.name}: window {cfg['window']} khac pair_params {self.pair['window']}")
        self.mean = np.asarray(cfg["mean"], np.float32)
        self.std = np.asarray(cfg["std"], np.float32)
        self.threshold = float(cfg.get("threshold", self.threshold))
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print(f"[fight] ONNX {model.name} | nguong {self.threshold} | pair_params {cfg.get('pair_level', '')}")

    # ------------------------------------------------------------ chấm điểm cặp
    def _onnx_scores(self, inst):
        As = {k: np.stack([i[3][k] for i in inst]) for k in ("kpts", "box", "valid")}
        Bs = {k: np.stack([i[4][k] for i in inst]) for k in ("kpts", "box", "valid")}
        x = ((F.pair_features(As, Bs) - self.mean) / self.std).astype(np.float32)
        logit = self.session.run(None, {self.input_name: x})[0].reshape(-1)
        return _sigmoid(logit)

    def _fake_scores(self, inst):
        """1 nếu (khoảng cách tâm TB < fake_dist × chiều cao) và (tốc độ cổ tay TB > fake_wrist_speed chiều cao/s)
        trong 1 s cuối cửa sổ, ngược lại 0."""
        n = self.fps
        out = []
        for a, b, _, A, B in inst:
            v = A["valid"][-n:] & B["valid"][-n:]
            if v.sum() < n // 2:
                out.append(0.0)
                continue
            ca = (A["box"][-n:, :2] + A["box"][-n:, 2:]) / 2
            cb = (B["box"][-n:, :2] + B["box"][-n:, 2:]) / 2
            h = ((A["box"][-n:, 3] - A["box"][-n:, 1]) + (B["box"][-n:, 3] - B["box"][-n:, 1])) / 2
            dist = float((np.linalg.norm(ca - cb, axis=1)[v] / np.maximum(h[v], 1)).mean())
            hm = float(h[v].mean())
            speed = max(self._wrist_speed(A, n) / hm, self._wrist_speed(B, n) / hm)
            self.fake_dbg[(a, b)] = (dist, speed)
            out.append(float(dist < self.fake_dist and speed > self.fake_speed))
        return np.array(out, np.float32)

    def _wrist_speed(self, S, n):
        """Tốc độ TB (pixel / giây) của cổ tay NHANH HƠN trong n frame cuối (đấm thường 1 tay; lấy TB cả 2 tay
        sẽ chia đôi tốc độ). Chỉ tính các bước có keypoint tin cậy."""
        k, valid = S["kpts"][-n:], S["valid"][-n:]
        ok = (k[:, WRISTS, 2] >= F.KPT_CONF) & valid[:, None]
        step = ok[1:] & ok[:-1]                                    # (n-1, 2)
        d = np.linalg.norm(k[1:, WRISTS, :2] - k[:-1, WRISTS, :2], axis=-1)
        speeds = [d[step[:, j], j].mean() for j in range(2) if step[:, j].sum() >= n // 3]
        return float(max(speeds) * self.fps) if speeds else 0.0

    # ------------------------------------------------------------ vòng chính
    def update(self, frame, tracks, t):
        self.tick += 1
        for tid in self.gone_ids(tracks, t):
            self.buf.pop(tid, None)
        for tid, tr in tracks.items():
            self.buf.setdefault(tid, deque(maxlen=self.window)).append((self.tick, tr["box"], tr["kpts"]))
        alive = set(self.buf)
        for key in [k for k in self.ema if not set(k) <= alive]:
            self._drop(key)
        if self.tick % self.infer_every:
            return []

        start = self.tick - self.window + 1
        tracks_k = {tid: {k: (b, kp) for k, b, kp in q if k >= start} for tid, q in self.buf.items()}
        tracks_k = {tid: d for tid, d in tracks_k.items() if d}
        inst = F.build_pair_instances(tracks_k, starts=[start], **self.pair) if len(tracks_k) >= 2 else []
        probs = (self._onnx_scores(inst) if self.session is not None else self._fake_scores(inst)) if inst else []
        seen = set()
        for (a, b, *_), p in zip(inst, probs):
            key = (a, b)
            seen.add(key)
            self.last_p[key] = float(p)
            self.ema[key] = self.alpha * float(p) + (1 - self.alpha) * self.ema.get(key, 0.0)
        for key in list(self.ema):                              # cặp không còn là ứng viên: EMA giảm dần
            if key not in seen:
                self.last_p[key] = 0.0
                self.ema[key] *= 1 - self.alpha
                if self.ema[key] < 0.01:
                    self._drop(key)

        events = []
        for key, e in self.ema.items():
            if e > self.threshold:
                t0 = self.above.setdefault(key, t)
                if t - t0 >= self.hold and key not in self.fired:
                    self.fired.add(key)
                    events.append(self.event("ĐÁNH NHAU", key, round(e, 2)))
            else:
                self.above.pop(key, None)
                self.fired.discard(key)
        return events

    def _drop(self, key):
        for d in (self.ema, self.last_p, self.above, self.fake_dbg):
            d.pop(key, None)
        self.fired.discard(key)

    def highlight_ids(self):
        """Cặp đã xác nhận đánh nhau và EMA vẫn > ngưỡng."""
        return {i for key in self.fired if self.ema.get(key, 0) > self.threshold for i in key}

    def debug_info(self, tid):
        lines = []
        for (a, b), e in self.ema.items():
            if tid not in (a, b):
                continue
            other = b if tid == a else a
            held = self.above.get((a, b))
            s = f"DANH NHAU [{self.mode}] voi {other}: p={self.last_p.get((a, b), 0):.2f} ema={e:.2f}/{self.threshold}"
            if (a, b) in self.fake_dbg:
                dist, speed = self.fake_dbg[(a, b)]
                s += f" kc={dist:.2f}/{self.fake_dist} tay={speed:.1f}/{self.fake_speed}"
            lines.append(s + (" GIU" if held is not None else ""))
        return lines

    def debug_scene(self):
        return [f"Đánh nhau: {self.mode}, {len(self.ema)} cặp đang theo dõi, tick {self.tick}"]
