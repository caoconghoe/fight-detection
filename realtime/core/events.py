"""EventManager: gọi các detector mỗi tick lấy mẫu, gộp – lọc – ưu tiên sự kiện (CLAUDE.md mục 5.4)."""
import importlib

PRIORITY = {"ĐÁNH NHAU": 100, "TÉ NGÃ RỒI NẰM BẤT ĐỘNG": 90, "HOẢNG LOẠN": 85,
            "TÉ NGÃ": 80, "GIƠ TAY ĐẦU HÀNG": 70, "VẪY TAY CẦU CỨU": 60,
            "NẰM BẤT ĐỘNG": 60, "CHẠY NHANH": 55, "CẢNH BẤT THƯỜNG": 50, "ÔM ĐẦU / GỤC ĐẦU": 40}

FIGHT, FALL, STILL, FALL_STILL = "ĐÁNH NHAU", "TÉ NGÃ", "NẰM BẤT ĐỘNG", "TÉ NGÃ RỒI NẰM BẤT ĐỘNG"
PANIC, RUN = "HOẢNG LOẠN", "CHẠY NHANH"

# tên trong config.yaml -> "module:Lớp"
REGISTRY = {"fight": "detectors.fight:FightDetector",
            "fall": "detectors.fall:FallDetector",
            "pose_rules": "detectors.pose_rules:PoseRulesDetector",
            "panic": "detectors.panic:PanicDetector",
            "running": "detectors.running:RunningDetector",
            "scene": "detectors.scene:SceneDetector"}


def build_detectors(cfg):
    """Tạo các detector được bật trong cfg["detectors"]. Detector chưa viết thì bỏ qua (in cảnh báo)."""
    dets = []
    for name, params in (cfg.get("detectors") or {}).items():
        params = dict(params or {})
        if not params.pop("enabled", True):
            continue
        mod_name, cls_name = REGISTRY[name].split(":")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            if e.name != mod_name:
                raise
            print(f"[canh bao] chua co detector '{name}' ({mod_name.replace('.', '/')}.py) -> bo qua")
            continue
        dets.append(getattr(mod, cls_name)(fps=cfg["sample_fps"], full_body=cfg.get("full_body"), **params))
    return dets


class EventManager:
    def __init__(self, detectors, cooldown_sec=10.0, fight_hold_sec=3.0, fall_upgrade_sec=30.0, panic_hold_sec=10.0):
        self.detectors = list(detectors)
        self.cooldown, self.fight_hold, self.fall_upgrade = cooldown_sec, fight_hold_sec, fall_upgrade_sec
        self.panic_hold = panic_hold_sec
        self.last_panic = -1e9   # t lần cuối có HOẢNG LOẠN (kể cả khi bị cooldown)
        for d in self.detectors:
            getattr(d, "attach", lambda ds: None)(self.detectors)
        self.last_fight = {}     # id -> t lần cuối nằm trong ĐÁNH NHAU
        self.last_fall = {}      # id -> t lần cuối TÉ NGÃ
        self.last_sent = {}      # (type, frozenset(ids)) -> t lần cuối đã báo

    def update(self, frame, tracks, t):
        """Gọi mỗi tick lấy mẫu. Trả về (list sự kiện cuối cùng, tập ID cần tô đỏ)."""
        raw = []
        for d in self.detectors:
            for e in d.update(frame, tracks, t) or []:
                e = dict(e)
                e["ids"] = [int(i) for i in e.get("ids", [])]
                e.setdefault("severity", PRIORITY.get(e["type"], 0))
                e.setdefault("confidence", None)
                raw.append(e)

        # ghi nhớ té ngã / đánh nhau (kể cả khi sự kiện bị cooldown) để nâng cấp và ưu tiên
        for e in raw:
            if e["type"] == FALL:
                self.last_fall.update({i: t for i in e["ids"]})
            elif e["type"] == FIGHT:
                self.last_fight.update({i: t for i in e["ids"]})
            elif e["type"] == PANIC:
                self.last_panic = t

        # HOẢNG LOẠN đang báo (cùng tick hoặc ≤ panic_hold_sec) -> không báo CHẠY NHANH của bất kỳ ai
        if t - self.last_panic <= self.panic_hold:
            raw = [e for e in raw if e["type"] != RUN]

        # nâng cấp: NẰM BẤT ĐỘNG sau TÉ NGÃ (≤ 30 s) -> TÉ NGÃ RỒI NẰM BẤT ĐỘNG
        for e in raw:
            if e["type"] == STILL and any(t - self.last_fall.get(i, -1e9) <= self.fall_upgrade
                                          for i in e["ids"]):
                e["type"], e["severity"] = FALL_STILL, PRIORITY[FALL_STILL]

        # ưu tiên: ID đã thuộc sự kiện cao hơn (cùng tick hoặc đang đánh nhau ≤ 3 s) -> bỏ sự kiện thấp hơn
        claimed = {i: PRIORITY[FIGHT] for i, tf in self.last_fight.items() if t - tf <= self.fight_hold}
        kept = []
        for e in sorted(raw, key=lambda e: -e["severity"]):
            if any(claimed.get(i, -1) > e["severity"] for i in e["ids"]):
                continue
            for i in e["ids"]:
                claimed[i] = max(claimed.get(i, -1), e["severity"])
            kept.append(e)

        # cooldown: cùng (type, tập ids) không báo lại trong 10 s
        events = []
        for e in kept:
            key = (e["type"], frozenset(e["ids"]))
            if t - self.last_sent.get(key, -1e9) < self.cooldown:
                continue
            self.last_sent[key] = t
            events.append(e)

        self._prune(t)
        highlight = set()
        for d in self.detectors:
            highlight |= set(d.highlight_ids())
        return events, highlight

    def debug(self, tracks):
        """(chế độ debug) -> ({tid: [dòng chữ]}, [dòng chữ cho cả cảnh]) từ mọi detector."""
        per_id = {tid: [line for d in self.detectors for line in d.debug_info(tid)] for tid in tracks}
        scene = [line for d in self.detectors for line in d.debug_scene()]
        return per_id, scene

    def _prune(self, t):
        self.last_fight = {i: v for i, v in self.last_fight.items() if t - v <= self.fight_hold}
        self.last_fall = {i: v for i, v in self.last_fall.items() if t - v <= self.fall_upgrade}
        self.last_sent = {k: v for k, v in self.last_sent.items() if t - v < self.cooldown}
