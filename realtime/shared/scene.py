"""LUỒNG 2 – nhìn ảnh màu cả cảnh (UCF-Crime). CLAUDE.md mục 12.

DÙNG CHUNG giữa notebook Kaggle (04 trích đặc trưng, 05 train) và realtime (detectors/scene.py):
- hằng số (2 FPS, 224 px, đoạn 16 frame), cách nhận LỚP từ đường dẫn video, ánh xạ lớp -> NHÓM;
- tiền xử lý 1 frame: BGR -> resize 224×224 (ép cả khung, không cắt) -> RGB uint8. Chuẩn hoá ImageNet nằm
  TRONG mô hình backbone (scene_backbone.onnx nhận uint8), nên realtime và lúc train không thể lệch nhau.
Chỉ phụ thuộc numpy + cv2.
"""
import re

import numpy as np

FPS = 2                     # lấy mẫu 2 frame / giây theo thời gian (shared/sampling.py)
SIZE = 224
SEGMENT = 16                # 16 frame mẫu = 8 s / đoạn
FEAT_DIM = 960              # MobileNetV3-Large: vector sau global average pooling
MAX_SEC = 15 * 60           # chỉ lấy 15 phút đầu mỗi video
VIDEO_EXT = {".mp4", ".avi", ".mpg", ".mpeg", ".mkv", ".mov", ".m4v", ".webm", ".wmv"}
SKIP_PATH = ("/anomaly/",)  # bản UCF mission-ai có thư mục /anomaly/ trùng -> bỏ
LOW_PRIORITY_DUP = ("real life violence situations",)   # bản sao trùng của RLVS -> chỉ giữ khi không có bản chính
NORMAL = "Normal"           # mọi thư mục / tên file có chữ "normal" -> lớp Normal
SOURCE_OF = {"Fight": "rwf", "NonFight": "rwf", "Violence": "rlvs", "NonViolence": "rlvs"}   # còn lại: ucf


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def class_index(groups, eval_only=None):
    """groups: {nhóm: [lớp, ...]} (config.yaml scene.groups); eval_only: {lớp: nhóm} chỉ dùng đánh giá.
    -> {tên chuẩn hoá: (Lớp, nhóm, dùng để train?)}"""
    idx = {}
    for g, classes in groups.items():
        for c in classes:
            idx[norm(c)] = (c, g, True)
    for c, g in (eval_only or {}).items():
        idx[norm(c)] = (c, g, False)
    return idx


def detect_class(path, index):
    """Lớp của video từ đường dẫn: thư mục cha gần nhất trùng tên lớp (không phân biệt hoa thường, bỏ dấu câu);
    thư mục có chữ "normal" -> Normal; nếu không thấy, thử tiền tố tên file (vd Abuse001_x264.mp4). None = bỏ."""
    parts = str(path).replace("\\", "/").split("/")
    for comp in reversed(parts[:-1]):
        n = norm(comp)
        if n in index:
            return index[n][0]
        if "normal" in n and norm(NORMAL) in index:
            return NORMAL
    name = parts[-1]
    m = re.match(r"[A-Za-z]+", name)
    if m and norm(m.group(0)) in index:
        return index[norm(m.group(0))][0]
    if "normal" in norm(name) and norm(NORMAL) in index:
        return NORMAL
    return None


def source_of(cls):
    return SOURCE_OF.get(cls, "ucf")


def skip_path(path):
    p = str(path).replace("\\", "/").lower()
    return any(s in p for s in SKIP_PATH)


def dup_rank(path):
    """Thứ tự ưu tiên khi trùng tên file: 0 = bản chính, 1 = bản sao đã biết (giữ bản có rank nhỏ hơn)."""
    p = str(path).replace("\\", "/").lower()
    return int(any(s in p for s in LOW_PRIORITY_DUP))


def video_key(source, cls, path):
    """Khoá duy nhất (tên file .npy) – chống trùng theo (nguồn, lớp, tên file)."""
    stem = str(path).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{source}__{cls}__{stem}")


def prep(frame_bgr):
    """1 frame BGR (bất kỳ kích thước) -> RGB uint8 (224, 224, 3), ép cả khung (không cắt mất mép cảnh)."""
    import cv2
    small = cv2.resize(frame_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def segments(n, segment=SEGMENT, stride=None):
    """Điểm bắt đầu các đoạn `segment` frame mẫu phủ n frame (đoạn cuối căn về cuối; n < segment -> [0])."""
    stride = stride or segment
    if n <= segment:
        return [0]
    starts = list(range(0, n - segment + 1, stride))
    if starts[-1] != n - segment:
        starts.append(n - segment)
    return starts


def parse_temporal_annotation(text):
    """Temporal_Anomaly_Annotation_for_Testing_Videos.txt -> {tên file: (lớp, [(frame_đầu, frame_cuối), ...])}.
    Mỗi dòng: <tên> <lớp> s1 e1 s2 e2 (frame ở FPS gốc, -1 = không có)."""
    out = {}
    for line in text.splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        nums = [int(float(x)) for x in p[2:6]]
        spans = [(s, e) for s, e in (nums[0:2], nums[2:4]) if s >= 0 and e >= 0]
        out[p[0]] = (p[1], spans)
    return out


def parse_test_list(text):
    """Anomaly_Test.txt (vd "Abuse/Abuse028_x264.mp4") -> tập tên file."""
    return {ln.strip().replace("\\", "/").split("/")[-1] for ln in text.splitlines() if ln.strip()}


def frame_labels(times, fps_src, spans):
    """Nhãn bất thường (0/1) cho các frame mẫu tại thời điểm `times` (giây) theo khoảng chú thích (frame gốc)."""
    t = np.asarray(times, float)
    y = np.zeros(len(t), np.int8)
    for s, e in spans:
        y[(t >= s / fps_src) & (t <= e / fps_src)] = 1
    return y
