"""LUỒNG 2 – CẢNH BẤT THƯỜNG: nhìn ảnh màu cả cảnh (CLAUDE.md mục 12).

- update() (gọi ở mỗi tick 15 FPS) chỉ lấy frame 2 FPS và đẩy vào hàng đợi, KHÔNG suy luận ở luồng chính.
- Luồng nền: scene.prep (224 px RGB uint8) → backbone MobileNetV3 (OpenVINO; không được thì ONNX Runtime)
  → chuẩn hoá mean/std trong scene_config.json → giữ 16 vector gần nhất (~8 s) → scene_head.onnx → điểm cửa sổ + logit nhóm.
- Cảnh báo khi điểm cửa sổ > threshold ở ít nhất min_consecutive cửa sổ liên tiếp. Tên nhóm chỉ là GỢI Ý (phân nhóm còn ~40%).
"""
import json
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np

from detectors.base import Detector
from shared import scene as S
from shared.sampling import FpsSampler

ROOT = Path(__file__).resolve().parents[1]                  # thư mục realtime/
MODEL_DIR = ROOT / "models"


class SceneScorer:
    """Backbone + head. Dùng lại được cho detector và cho tools/calibrate_scene.py."""

    def __init__(self, model_dir=MODEL_DIR, threads=2, backend="openvino"):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "scene_config.json").read_text(encoding="utf-8"))
        self.mean = np.asarray(self.cfg["mean"], np.float32)
        self.std = np.asarray(self.cfg["std"], np.float32)
        self.win = int(self.cfg["window"])
        self.groups = list(self.cfg["groups_abnormal"])
        self.backbone, self.backend = self._load_backbone(model_dir / "scene_backbone.onnx", threads, backend)

        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.head = ort.InferenceSession(str(model_dir / "scene_head.onnx"), so, providers=["CPUExecutionProvider"])
        self.head_in = self.head.get_inputs()[0].name

        self.buf = deque(maxlen=self.win)
        self.n_in = 0                                       # số frame 2 FPS đã nhận
        self.n_dropped = 0                                  # số frame bị bỏ (chỉ khi máy quá chậm, nguồn camera)
        self.q = queue.Queue(maxsize=64)
        self.out = deque(maxlen=256)                        # (t, điểm cửa sổ, logit nhóm)
        self.lock = threading.Lock()
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    @staticmethod
    def _load_backbone(path, threads, backend):
        if backend == "openvino":
            try:
                import openvino as ov
                core = ov.Core()
                m = core.read_model(str(path))
                m.reshape({m.input(0): [1, S.SIZE, S.SIZE, 3]})
                cm = core.compile_model(m, "CPU", {"INFERENCE_NUM_THREADS": str(threads)})
                return (lambda x: cm([x])[cm.output(0)]), "openvino"
            except Exception as e:                          # không dùng được OpenVINO -> ONNX Runtime
                print(f"[scene] OpenVINO khong dung duoc ({type(e).__name__}: {e}) -> dung ONNX Runtime")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        return (lambda x: sess.run(None, {name: x})[0]), "onnxruntime"

    def submit(self, t, frame_bgr):
        """Đẩy 1 frame (BGR) vào hàng đợi. KHÔNG bỏ frame để giữ lưới thời gian đều (video, cửa sổ phải liền mạch).
        Chỉ khi hàng đợi đầy quá 2 s (máy quá chậm) mới bỏ frame cũ nhất và đếm lại."""
        item = (t, S.prep(frame_bgr))
        self.n_in += 1
        try:
            self.q.put(item, timeout=2.0)
        except queue.Full:
            self.n_dropped += 1
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(item)

    def drain(self):
        with self.lock:
            items = list(self.out)
            self.out.clear()
        return items

    def _loop(self):
        while self.running:
            try:
                t, rgb = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            feat = np.asarray(self.backbone(rgb[None]), np.float32).reshape(-1)
            self.buf.append((feat - self.mean) / self.std)
            if len(self.buf) == self.win:
                x = np.stack(self.buf)[None].astype(np.float32)
                s, g = self.head.run(None, {self.head_in: x})
                with self.lock:
                    self.out.append((t, float(np.ravel(s)[0]), np.ravel(g)))

    def close(self):
        self.running = False


class SceneDetector(Detector):
    name = "scene"

    def __init__(self, fps=15, threshold=-0.80, min_consecutive=2, threads=2, backend="openvino",
                 model_dir=None, group_note="~40%", scorer=None, **kw):
        super().__init__(fps, **kw)
        self.threshold, self.min_consec, self.group_note = threshold, min_consecutive, group_note
        self.sampler = FpsSampler(S.FPS)                    # 2 FPS theo thời gian
        self.scorer = scorer
        if self.scorer is None:
            try:
                self.scorer = SceneScorer(Path(model_dir) if model_dir else MODEL_DIR, threads, backend)
                print(f"[scene] DUNG LUONG 2 | backend {self.scorer.backend} | nguong {threshold} | "
                      f"{min_consecutive} cua so lien tiep | cua so {self.scorer.win} frame")
            except (FileNotFoundError, OSError) as e:
                print(f"[scene] chua co mo hinh luong 2 ({e}) -> TAT luong 2")
        self.run, self.fired, self.run_groups = 0, False, []
        self.last_score, self.last_group, self.n_windows = None, None, 0
        self.history = []                                   # (t_khung_cuoi, điểm, nhóm gợi ý) cho mọi cửa sổ

    def update(self, frame, tracks, t):
        if self.scorer is None:
            return []
        if frame is not None and self.sampler.tick(t):
            self.scorer.submit(t, frame)
        events = []
        for ts, s, g in self.scorer.drain():
            self.n_windows += 1
            self.last_score = s
            self.last_group = self.scorer.groups[int(np.argmax(g))]
            self.history.append((ts, s, self.last_group))
            if s > self.threshold:
                self.run += 1
                self.run_groups.append(g)
                if self.run >= self.min_consec and not self.fired:
                    self.fired = True
                    grp = self.scorer.groups[int(np.argmax(np.mean(self.run_groups, 0)))]
                    events.append(self.event("CẢNH BẤT THƯỜNG", [], None, scene_score=round(s, 3), scene_group=grp,
                                             detail=f"gợi ý: {grp} (phân nhóm đúng {self.group_note})"))
            else:
                self.run, self.fired, self.run_groups = 0, False, []
        return events

    def debug_scene(self):
        if self.scorer is None:
            return ["Canh bat thuong: TAT (chua co mo hinh)"]
        sc = "-" if self.last_score is None else f"{self.last_score:.2f}"
        return [f"Canh bat thuong [{self.scorer.backend}]: diem {sc} / nguong {self.threshold}",
                f"  cua so: {self.n_windows} | chuoi vuot nguong: {self.run}/{self.min_consec}"]

    def close(self):
        if self.scorer is None:
            return
        if self.n_windows == 0:
            need = self.scorer.win
            print(f"[scene] CANH BAO: khong co cua so nao. Da nhan {self.scorer.n_in} khung 2 FPS, can {need} khung "
                  f"(~{need / 2:.0f} s) moi co 1 cua so. Video qua ngan -> luong 2 khong the bao.")
        if self.scorer.n_dropped:
            print(f"[scene] CANH BAO: da bo {self.scorer.n_dropped} khung (may qua cham)")
        self.scorer.close()
