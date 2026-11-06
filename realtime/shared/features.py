"""Đặc trưng CẶP NGƯỜI cho mô hình ĐÁNH NHAU (Pair-TCN) – CLAUDE.md mục 7.1.

DÙNG CHUNG giữa notebook Kaggle (train) và realtime (detectors/fight.py) -> chỉ dùng numpy.
Mọi thứ tính trên trục thời gian ĐÃ LẤY MẪU về 15 FPS: frame mẫu thứ k ứng với thời điểm ≈ k / 15 s,
nên không đặc trưng nào phụ thuộc FPS gốc hay số frame của video.
Tiêu chí ghép cặp nằm DUY NHẤT trong PARAMS (realtime và Kaggle dùng chung; áp dụng như nhau mọi nhãn).

Dạng dữ liệu:
- tracks (theo thời gian thật):  {tid: [(t_giây, box_px(4,), kpts_px(17, 3)), ...]}
- tracks_k (đã lấy mẫu 15 FPS):  {tid: {k: (box_px(4,), kpts_px(17, 3))}}
- seq (1 người trong 1 cửa sổ):  {"box": (..., W, 4), "kpts": (..., W, 17, 3), "valid": (..., W) bool,
                                  "present": (..., W) bool}   (valid = có dữ liệu sau khi lấp frame thiếu)
"""
import bisect

import numpy as np

try:
    from shared.sampling import sample_indices          # realtime (chạy từ thư mục realtime/)
except ImportError:
    from sampling import sample_indices                 # Kaggle (%%writefile sampling.py)

FPS = 15
WINDOW = 32                 # frame mẫu / cửa sổ (~2.1 s)
STRIDE = 8
FEAT_DIM = 180
KPT_CONF = 0.3              # keypoint conf < ngưỡng -> toạ độ 0, conf 0
FAR = 3.0                   # khoảng cách khi thiếu dữ liệu (và giới hạn trên), đơn vị chiều cao người
VEL_CLIP = 1.0              # chặn vận tốc (chiều cao / frame mẫu) – bỏ nhảy do đổi ID

# Tiêu chí ghép cặp / dựng cửa sổ – MỘT NGUỒN DUY NHẤT (truyền **kw để thử mức khác, vd khi chẩn đoán)
PARAMS = dict(
    window=WINDOW, stride=STRIDE,
    min_copresence=0.4,     # 2 người cùng có mặt (sau khi lấp) ≥ 40% số frame của cửa sổ
    max_fill=8,             # lấp (nội suy) khoảng trống ≤ 8 frame mẫu (~0.5 s)
    near_ratio=2.0,         # ≥ 1 frame có khoảng cách tâm box ≤ 2.0 × chiều cao TB của 2 người
    max_pairs=6,            # tối đa 6 cặp gần nhau nhất / cửa sổ
    tail_window=True,       # thêm cửa sổ cuối căn về cuối clip (không bỏ sót đoạn cuối)
    pad_short=True,         # clip < 1 cửa sổ: vẫn tạo 1 cửa sổ, phần thiếu đệm 0 (conf 0)
    assign_iou=0.5,         # offline: detection track_id = -1 -> gán vào track gần nhất nếu IoU > 0.5 (None = bỏ)
    link_gap_sec=0.5,       # nối tracklet: B bắt đầu ≤ 0.5 s sau khi A kết thúc...
    link_dist=0.5,          # ...và tâm box cách nhau < 0.5 × chiều cao box của A
)

HEAD = [0, 1, 2, 3, 4]
TORSO = [5, 6, 11, 12]
L_WR, R_WR, L_HIP, R_HIP = 9, 10, 11, 12
FLIP_KPTS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # hoán đổi trái/phải

# Bố cục 180 đặc trưng mỗi frame
SL_XY = slice(0, 68)        # A: 17×(x, y) -> 0..33, B -> 34..67  (gốc = trung điểm hông A, B; chia thang đo)
SL_CONF = slice(68, 102)    # A: 68..84, B: 85..101
SL_VEL = slice(102, 170)    # A: 102..135, B: 136..169 (hiệu 2 frame mẫu liên tiếp, đã chuẩn hoá)
SL_WRIST = slice(170, 178)  # A_Ltay→B_đầu, A_Ltay→B_thân, A_Rtay→B_đầu, A_Rtay→B_thân, rồi B→A tương tự
SL_DIST = slice(178, 180)   # khoảng cách tâm A–B, tốc độ thay đổi của nó


def _p(kw):
    unknown = set(kw) - set(PARAMS)
    if unknown:
        raise TypeError(f"Tham so la: {sorted(unknown)}")
    return {**PARAMS, **kw}


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, iw) * max(0.0, ih)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- dữ liệu offline (npz)

def tracks_from_npz(npz, assign_iou=PARAMS["assign_iou"], max_gap_sec=PARAMS["link_gap_sec"]):
    """npz của notebook 01_extract_pose -> {tid: [(t, box_px, kpts_px)]} ở FPS GỐC.
    Toạ độ trong npz chuẩn hoá 0..1 theo W và H riêng -> nhân lại về pixel (tránh méo tỉ lệ).
    Detection track_id = -1 (tracker chưa gán): gán vào track (chưa có detection ở frame đó) có box ở frame
    gần nhất (≤ max_gap_sec) với IoU > assign_iou lớn nhất; không có thì bỏ. assign_iou=None -> bỏ hết."""
    data = np.asarray(npz["data"], np.float32).reshape(-1, 57)
    fps, W, H = float(npz["fps"]), float(npz["width"]), float(npz["height"])
    scale = np.array([W, H, W, H], np.float32)

    def unpack(row):
        k = row[6:].reshape(17, 3).copy()
        k[:, 0] *= W
        k[:, 1] *= H
        return row[2:6] * scale, k

    tracks = {}                                          # tid -> {frame: (box, kpts)}
    known = data[data[:, 1] >= 0]
    for row in known[np.lexsort((known[:, 0], known[:, 1]))]:
        tracks.setdefault(int(row[1]), {})[int(row[0])] = unpack(row)
    unk = data[data[:, 1] < 0]
    if assign_iou is not None and len(unk) and tracks:
        frames = {tid: sorted(d) for tid, d in tracks.items()}
        max_gap = max(1, int(round(max_gap_sec * fps)))
        for row in unk[np.argsort(unk[:, 0], kind="stable")]:
            f = int(row[0])
            box, k = unpack(row)
            best, best_iou = None, assign_iou
            for tid, d in tracks.items():
                if f in d:
                    continue
                fl = frames[tid]
                i = bisect.bisect_left(fl, f)
                near = [fl[j] for j in (i - 1, i) if 0 <= j < len(fl) and abs(fl[j] - f) <= max_gap]
                if not near:
                    continue
                v = _iou(box, d[min(near, key=lambda x: abs(x - f))][0])
                if v > best_iou:
                    best, best_iou = tid, v
            if best is not None:
                tracks[best][f] = (box, k)
                bisect.insort(frames[best], f)
    return {tid: [(f / fps, *d[f]) for f in sorted(d)] for tid, d in tracks.items()}


def clip_times(npz):
    """Thời điểm của mọi frame trong clip (kể cả frame không có ai) – lưới để lấy mẫu."""
    return np.arange(int(npz["n_frames"])) / float(npz["fps"])


def sample_tracks(tracks, fps=FPS, times=None):
    """Lấy mẫu về fps theo THỜI GIAN bằng logic FpsSampler trên lưới thời gian CHUNG của clip
    (mọi người cùng các frame mẫu). Frame mẫu tại thời điểm t có chỉ số k = round(t × fps).
    times: thời điểm mọi frame của clip (clip_times); None = hợp các thời điểm có người."""
    if times is None:
        times = sorted({t for tr in tracks.values() for t, _, _ in tr})
    times = np.asarray(times, float)
    keep = {round(float(times[i]), 6) for i in sample_indices(list(times), fps)}
    out = {}
    for tid, tr in tracks.items():
        d = {}
        for t, box, k in tr:
            if round(t, 6) in keep:
                d.setdefault(int(round(t * fps)), (np.asarray(box, np.float32), np.asarray(k, np.float32)))
        if d:
            out[tid] = d
    return out


def _center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def link_tracklets(tracks_k, fps=FPS, max_gap_sec=PARAMS["link_gap_sec"], max_dist=PARAMS["link_dist"]):
    """Nối tracklet (offline, vì ByteTrack persist=False hay đổi ID): track A kết thúc, track B bắt đầu
    sau đó ≤ max_gap_sec giây và tâm box cách nhau < max_dist × chiều cao box của A -> gộp B vào A.
    Mỗi B chọn A gần nhất. Trả về dict mới {tid: {k: (box, kpts)}}."""
    merged = {tid: dict(d) for tid, d in tracks_k.items()}
    heads = {}                                   # tid chuỗi -> (k cuối, box cuối)
    for s, e, tid in sorted((min(d), max(d), tid) for tid, d in tracks_k.items()):
        first_box = merged[tid][s][0]
        best = None
        for a, (ae, abox) in heads.items():
            gap = s - ae
            if 0 < gap <= max_gap_sec * fps + 1e-9:
                dist = np.linalg.norm(_center(first_box) - _center(abox))
                if dist < max_dist * (abox[3] - abox[1]) and (best is None or dist < best[0]):
                    best = (dist, a)
        if best is None:
            heads[tid] = (e, merged[tid][e][0])
        else:
            a = best[1]
            merged[a].update(merged.pop(tid))
            heads[a] = (e, merged[a][e][0])
    return merged


# ---------------------------------------------------------------- cửa sổ, cặp

def fill_gaps(box, kpts, present, max_fill=PARAMS["max_fill"]):
    """Lấp frame thiếu của 1 người trong cửa sổ (sửa tại chỗ): khoảng trống ≤ max_fill frame giữa 2 frame
    có dữ liệu -> nội suy tuyến tính (conf = min 2 đầu); còn lại để 0, conf 0. Trả về mask valid."""
    valid = present.copy()
    idx = np.flatnonzero(present)
    for i, j in zip(idx[:-1], idx[1:]):
        if 1 < j - i <= max_fill + 1:
            for m in range(i + 1, j):
                w = (m - i) / (j - i)
                box[m] = (1 - w) * box[i] + w * box[j]
                kpts[m, :, :2] = (1 - w) * kpts[i, :, :2] + w * kpts[j, :, :2]
                kpts[m, :, 2] = np.minimum(kpts[i, :, 2], kpts[j, :, 2])
                valid[m] = True
    return valid


def seq_from_track(track_k, start, window=WINDOW, max_fill=PARAMS["max_fill"]):
    """1 người (dict k -> (box, kpts)) trong cửa sổ [start, start + window) -> seq (đã lấp frame thiếu).
    Frame không có dữ liệu (kể cả phần đệm ngoài clip) = 0, conf 0, valid False."""
    box = np.zeros((window, 4), np.float32)
    kpts = np.zeros((window, 17, 3), np.float32)
    present = np.zeros(window, bool)
    for k, (b, kp) in track_k.items():
        i = k - start
        if 0 <= i < window:
            box[i], kpts[i], present[i] = b, kp, True
    valid = fill_gaps(box, kpts, present, max_fill)
    return {"box": box, "kpts": kpts, "valid": valid, "present": present}


def pair_stats(A, B, n_frames=None):
    """(tỉ lệ có mặt chung, khoảng cách tâm nhỏ nhất / chiều cao TB) của 2 seq trong cửa sổ.
    n_frames: số frame của cửa sổ thuộc clip (mẫu số tỉ lệ; clip ngắn được đệm) – None = cả cửa sổ."""
    both = A["valid"] & B["valid"]
    n = len(both) if n_frames is None else max(1, n_frames)
    if not both.any():
        return 0.0, np.inf
    ca = (A["box"][both, :2] + A["box"][both, 2:]) / 2
    cb = (B["box"][both, :2] + B["box"][both, 2:]) / 2
    h = ((A["box"][both, 3] - A["box"][both, 1]) + (B["box"][both, 3] - B["box"][both, 1])) / 2
    return both.sum() / n, float((np.linalg.norm(ca - cb, axis=1) / np.maximum(h, 1.0)).min())


def candidate_pairs(seqs, n_frames=None, **kw):
    """Cặp ứng viên trong cửa sổ: có mặt chung ≥ min_copresence; có ≥ 1 frame khoảng cách tâm
    ≤ near_ratio × chiều cao TB. Trả về ≤ max_pairs cặp (idA < idB) gần nhau nhất."""
    p = _p(kw)
    ids = sorted(seqs)
    scored = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            co, d = pair_stats(seqs[a], seqs[b], n_frames)
            if co >= p["min_copresence"] and d <= p["near_ratio"]:
                scored.append((d, a, b))
    return [(a, b) for _, a, b in sorted(scored)[:p["max_pairs"]]]


def window_starts(k0, k1, **kw):
    """Điểm bắt đầu các cửa sổ phủ đoạn [k0, k1]: trượt bước stride + cửa sổ cuối căn về cuối (tail_window).
    Đoạn ngắn hơn 1 cửa sổ -> [k0] nếu pad_short (phần thiếu đệm 0), ngược lại []."""
    p = _p(kw)
    W = p["window"]
    if k1 - k0 + 1 < W:
        return [k0] if p["pad_short"] else []
    starts = list(range(k0, k1 - W + 2, p["stride"]))
    if p["tail_window"] and starts[-1] != k1 - W + 1:
        starts.append(k1 - W + 1)
    return starts


def _window_seqs(tracks_k, s, W, max_fill):
    return {tid: seq_from_track(d, s, W, max_fill) for tid, d in tracks_k.items()
            if any(s <= k < s + W for k in d)}


def build_pair_instances(tracks_k, starts=None, **kw):
    """-> list[(idA, idB, start_k, seqA, seqB)] cho mọi cửa sổ (hoặc các cửa sổ bắt đầu tại `starts`,
    vd realtime chỉ xét cửa sổ mới nhất)."""
    p = _p(kw)
    if not tracks_k:
        return []
    k0 = min(min(d) for d in tracks_k.values())
    k1 = max(max(d) for d in tracks_k.values())
    if starts is None:
        starts = window_starts(k0, k1, **kw)
    W, out = p["window"], []
    for s in starts:
        seqs = _window_seqs(tracks_k, s, W, p["max_fill"])
        n_frames = min(W, k1 - s + 1) if p["pad_short"] else W
        for a, b in candidate_pairs(seqs, n_frames, **kw):
            out.append((a, b, s, seqs[a], seqs[b]))
    return out


def build_pair_windows(tracks_k, fps=FPS, starts=None, **kw):
    """-> list[(idA, idB, start_t, feats(window, 180))] (CLAUDE.md 7.1 mục 8)."""
    return [(a, b, s / fps, pair_features(A, B))
            for a, b, s, A, B in build_pair_instances(tracks_k, starts, **kw)]


def clip_tracks(npz, fps=FPS, **kw):
    """npz -> theo track (gán detection -1) -> lấy mẫu 15 FPS -> nối tracklet -> tracks_k."""
    p = _p(kw)
    tracks = tracks_from_npz(npz, p["assign_iou"], p["link_gap_sec"])
    return link_tracklets(sample_tracks(tracks, fps, clip_times(npz)), fps, p["link_gap_sec"], p["link_dist"])


def clip_instances(npz, fps=FPS, **kw):
    """Toàn bộ pipeline offline cho 1 clip -> (instances [(idA, idB, start_k, seqA, seqB)], tracks_k)."""
    tracks_k = clip_tracks(npz, fps, **kw)
    return build_pair_instances(tracks_k, **kw), tracks_k


def explain_no_pairs(tracks_k, n_samples, **kw):
    """Vì sao clip KHÔNG có instance (cùng tiêu chí với build_pair_instances). n_samples: số frame mẫu của clip.
    -> None (có instance) | "a_clip_ngan" (< 1 cửa sổ) | "b_khong_co_2_nguoi" (không frame nào có ≥ 2 người)
       | "c_khong_du_co_mat_chung" (có ≥ 2 người nhưng không cặp nào đủ thời gian có mặt chung trong cửa sổ)
       | "d_qua_xa" (đủ thời gian có mặt chung nhưng luôn quá xa nhau)."""
    p = _p(kw)
    W = p["window"]
    if build_pair_instances(tracks_k, **kw):
        return None
    if n_samples < W:
        return "a_clip_ngan"
    count = {}
    for d in tracks_k.values():
        for k in d:
            count[k] = count.get(k, 0) + 1
    if max(count.values(), default=0) < 2:
        return "b_khong_co_2_nguoi"
    k0, k1 = min(count), max(count)
    for s in window_starts(k0, k1, **kw):
        seqs = _window_seqs(tracks_k, s, W, p["max_fill"])
        n_frames = min(W, k1 - s + 1) if p["pad_short"] else W
        ids = sorted(seqs)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if pair_stats(seqs[a], seqs[b], n_frames)[0] >= p["min_copresence"]:
                    return "d_qua_xa"
    return "c_khong_du_co_mat_chung"


# ---------------------------------------------------------------- đặc trưng

def _masked_mean(pts, ok):
    """pts (..., n, 2), ok (..., n) -> (trung bình các điểm ok (..., 2), có ít nhất 1 điểm (...))"""
    n = ok.sum(-1)
    m = (pts * ok[..., None]).sum(-2) / np.maximum(n, 1)[..., None]
    return m, n > 0


def _person(seq):
    k = np.asarray(seq["kpts"], np.float32)
    box = np.asarray(seq["box"], np.float32)
    valid = np.asarray(seq["valid"], bool)
    ok = (k[..., 2] >= KPT_CONF) & valid[..., None]                 # (..., W, 17)
    center = (box[..., :2] + box[..., 2:]) / 2
    hip, has_hip = _masked_mean(k[..., [L_HIP, R_HIP], :2], ok[..., [L_HIP, R_HIP]])
    hip = np.where(has_hip[..., None], hip, center)                 # thiếu hông -> tâm box
    height = box[..., 3] - box[..., 1]
    return k, ok, valid, center, hip, height


def pair_features(seqA, seqB):
    """Đặc trưng mỗi frame của cặp (A, B) -> (..., W, 180) float32 (hỗ trợ batch ở các chiều đầu).
    Gốc toạ độ = trung điểm tâm hông A và B (thiếu hông -> tâm box) ở TỪNG frame; thang đo S = chiều cao box
    trung bình của 2 người trong cửa sổ. Vận tốc tính với gốc cố định (trung bình cửa sổ) để giữ chuyển động."""
    kA, okA, vA, cA, hipA, hA = _person(seqA)
    kB, okB, vB, cB, hipB, hB = _person(seqB)

    def mean_h(h, v):
        return (h * v).sum(-1) / np.maximum(v.sum(-1), 1)
    nA, nB = vA.any(-1), vB.any(-1)
    S = np.where(nA & nB, (mean_h(hA, vA) + mean_h(hB, vB)) / 2, np.where(nA, mean_h(hA, vA), mean_h(hB, vB)))
    S = np.maximum(S, 1.0)                                          # (...)
    S1, S3 = S[..., None], S[..., None, None, None]

    both = vA & vB
    org = np.where(both[..., None], (hipA + hipB) / 2,
                   np.where(vA[..., None], hipA, np.where(vB[..., None], hipB, 0.0)))   # (..., W, 2)
    anyv = vA | vB
    O = (org * anyv[..., None]).sum(-2) / np.maximum(anyv.sum(-1), 1)[..., None]     # (..., 2)

    def xy(k, ok):
        return np.where(ok[..., None], (k[..., :2] - org[..., None, :]) / S3, 0.0)

    def vel(k, ok):
        q = (k[..., :2] - O[..., None, None, :]) / S3
        v = np.zeros_like(q)
        m = ok[..., 1:, :] & ok[..., :-1, :]
        v[..., 1:, :, :] = np.where(m[..., None], q[..., 1:, :, :] - q[..., :-1, :, :], 0.0)
        return np.clip(v, -VEL_CLIP, VEL_CLIP)

    def dist(p, okp, target, has):
        d = np.linalg.norm(p - target, axis=-1) / S1
        return np.where(okp & has, np.minimum(d, FAR), FAR)

    def targets(k, ok, center, v):
        head, has_head = _masked_mean(k[..., HEAD, :2], ok[..., HEAD])
        torso, has_torso = _masked_mean(k[..., TORSO, :2], ok[..., TORSO])
        torso = np.where(has_torso[..., None], torso, center)
        return head, has_head, torso, has_torso | v

    hdB, hhB, toB, htB = targets(kB, okB, cB, vB)
    hdA, hhA, toA, htA = targets(kA, okA, cA, vA)
    wrist = []
    for k, ok, hd, hh, to, ht in ((kA, okA, hdB, hhB, toB, htB), (kB, okB, hdA, hhA, toA, htA)):
        for w in (L_WR, R_WR):
            wrist += [dist(k[..., w, :2], ok[..., w], hd, hh), dist(k[..., w, :2], ok[..., w], to, ht)]

    dc = np.where(both, np.minimum(np.linalg.norm(cA - cB, axis=-1) / S1, FAR), FAR)
    rate = np.zeros_like(dc)
    rate[..., 1:] = np.where(both[..., 1:] & both[..., :-1], dc[..., 1:] - dc[..., :-1], 0.0)

    W = kA.shape[-3]
    lead = kA.shape[:-3]
    flat = lambda a: a.reshape(*lead, W, -1)                       # noqa: E731
    feats = np.concatenate([
        flat(xy(kA, okA)), flat(xy(kB, okB)),
        np.where(okA, kA[..., 2], 0.0), np.where(okB, kB[..., 2], 0.0),
        flat(vel(kA, okA)), flat(vel(kB, okB)),
        np.stack(wrist, -1), dc[..., None], rate[..., None]], axis=-1)
    return feats.astype(np.float32)


# ---------------------------------------------------------------- đối xứng (kiểm tra / tăng cường)

def _swap_perm():
    a, b = np.arange(0, 34), np.arange(34, 68)
    ca, cb = np.arange(68, 85), np.arange(85, 102)
    va, vb = np.arange(102, 136), np.arange(136, 170)
    wa, wb = np.arange(170, 174), np.arange(174, 178)
    return np.concatenate([b, a, cb, ca, vb, va, wb, wa, [178, 179]])


def _flip_perm_sign():
    perm, sign = np.arange(FEAT_DIM), np.ones(FEAT_DIM, np.float32)
    for base in (0, 34, 102, 136):                                   # khối (x, y) × 17
        for j, fj in enumerate(FLIP_KPTS):
            perm[base + 2 * j] = base + 2 * fj
            perm[base + 2 * j + 1] = base + 2 * fj + 1
            sign[base + 2 * j] = -1.0                                # x -> -x
    for base in (68, 85):                                            # conf × 17
        for j, fj in enumerate(FLIP_KPTS):
            perm[base + j] = base + fj
    for base in (170, 174):                                          # (Ltay->đầu, Ltay->thân, Rtay->đầu, Rtay->thân)
        perm[base:base + 4] = [base + 2, base + 3, base, base + 1]
    return perm, sign


SWAP_PERM = _swap_perm()
FLIP_PERM, FLIP_SIGN = _flip_perm_sign()


def swap_ab(feats):
    """Đặc trưng của cặp (B, A) từ đặc trưng cặp (A, B)."""
    return feats[..., SWAP_PERM]


def flip_h(feats):
    """Đặc trưng của cặp đã lật ngang (x -> -x, hoán đổi khớp trái/phải)."""
    return feats[..., FLIP_PERM] * FLIP_SIGN
