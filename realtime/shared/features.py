"""Đặc trưng CẶP NGƯỜI cho mô hình ĐÁNH NHAU (Pair-TCN) – CLAUDE.md mục 7.1.

DÙNG CHUNG giữa notebook Kaggle (train) và realtime (detectors/fight.py) -> chỉ dùng numpy.
Mọi thứ tính trên trục thời gian ĐÃ LẤY MẪU về 15 FPS: frame mẫu thứ k ứng với thời điểm ≈ k / 15 s,
nên không đặc trưng nào phụ thuộc FPS gốc hay số frame của video.

Dạng dữ liệu:
- tracks (theo thời gian thật):  {tid: [(t_giây, box_px(4,), kpts_px(17, 3)), ...]}
- tracks_k (đã lấy mẫu 15 FPS):  {tid: {k: (box_px(4,), kpts_px(17, 3))}}
- seq (1 người trong 1 cửa sổ):  {"box": (..., W, 4), "kpts": (..., W, 17, 3), "valid": (..., W) bool,
                                  "present": (..., W) bool}   (valid = có dữ liệu sau khi lấp frame thiếu)
"""
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
MAX_FILL = 3                # lấp (nội suy) khoảng trống ≤ N frame mẫu
MIN_PRESENCE = 0.6          # mỗi người phải có mặt ≥ 60% frame của cửa sổ
NEAR_RATIO = 1.5            # ≥ 1 frame có khoảng cách tâm ≤ 1.5 × chiều cao TB
MAX_PAIRS = 6               # tối đa N cặp gần nhau nhất / cửa sổ
LINK_GAP_SEC = 0.5          # nối tracklet: B bắt đầu ≤ 0.5 s sau khi A kết thúc...
LINK_DIST = 0.5             # ...và tâm box cách nhau < 0.5 × chiều cao box
FAR = 3.0                   # khoảng cách khi thiếu dữ liệu (và giới hạn trên), đơn vị chiều cao người
VEL_CLIP = 1.0              # chặn vận tốc (chiều cao / frame mẫu) – bỏ nhảy do đổi ID

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


# ---------------------------------------------------------------- dữ liệu offline (npz)

def tracks_from_npz(npz):
    """npz của notebook 01_extract_pose -> {tid: [(t, box_px, kpts_px)]} ở FPS GỐC.
    Toạ độ trong npz chuẩn hoá 0..1 theo W và H riêng -> nhân lại về pixel (tránh méo tỉ lệ). Bỏ tid -1."""
    data = np.asarray(npz["data"], np.float32).reshape(-1, 57)
    fps, W, H = float(npz["fps"]), float(npz["width"]), float(npz["height"])
    tracks = {}
    data = data[data[:, 1] >= 0]
    for row in data[np.lexsort((data[:, 0], data[:, 1]))]:
        box = row[2:6] * np.array([W, H, W, H], np.float32)
        k = row[6:].reshape(17, 3).copy()
        k[:, 0] *= W
        k[:, 1] *= H
        tracks.setdefault(int(row[1]), []).append((float(row[0]) / fps, box, k))
    return tracks


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


def link_tracklets(tracks_k, fps=FPS, max_gap_sec=LINK_GAP_SEC, max_dist=LINK_DIST):
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

def fill_gaps(box, kpts, present, max_fill=MAX_FILL):
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


def seq_from_track(track_k, start, window=WINDOW):
    """1 người (dict k -> (box, kpts)) trong cửa sổ [start, start + window) -> seq (đã lấp frame thiếu)."""
    box = np.zeros((window, 4), np.float32)
    kpts = np.zeros((window, 17, 3), np.float32)
    present = np.zeros(window, bool)
    for k, (b, kp) in track_k.items():
        i = k - start
        if 0 <= i < window:
            box[i], kpts[i], present[i] = b, kp, True
    valid = fill_gaps(box, kpts, present)
    return {"box": box, "kpts": kpts, "valid": valid, "present": present}


def candidate_pairs(seqs, min_presence=MIN_PRESENCE, near=NEAR_RATIO, max_pairs=MAX_PAIRS):
    """Cặp ứng viên trong cửa sổ: cả 2 có mặt ≥ min_presence số frame; có ≥ 1 frame khoảng cách tâm box
    ≤ near × chiều cao TB của 2 người. Trả về ≤ max_pairs cặp (idA < idB) gần nhau nhất."""
    ids = sorted(t for t, s in seqs.items() if s["present"].mean() >= min_presence)
    scored = []
    for i, a in enumerate(ids):
        A = seqs[a]
        for b in ids[i + 1:]:
            B = seqs[b]
            both = A["valid"] & B["valid"]
            if not both.any():
                continue
            ca = (A["box"][both, :2] + A["box"][both, 2:]) / 2
            cb = (B["box"][both, :2] + B["box"][both, 2:]) / 2
            h = ((A["box"][both, 3] - A["box"][both, 1]) + (B["box"][both, 3] - B["box"][both, 1])) / 2
            d = (np.linalg.norm(ca - cb, axis=1) / np.maximum(h, 1.0)).min()
            if d <= near:
                scored.append((d, a, b))
    return [(a, b) for _, a, b in sorted(scored)[:max_pairs]]


def build_pair_instances(tracks_k, window=WINDOW, stride=STRIDE, starts=None):
    """-> list[(idA, idB, start_k, seqA, seqB)] cho mọi cửa sổ (hoặc các cửa sổ bắt đầu tại `starts`,
    vd realtime chỉ xét cửa sổ mới nhất). Clip ngắn hơn 1 cửa sổ -> 1 cửa sổ, phần thiếu coi như vắng."""
    if not tracks_k:
        return []
    if starts is None:
        k0 = min(min(d) for d in tracks_k.values())
        k1 = max(max(d) for d in tracks_k.values())
        starts = list(range(k0, k1 - window + 2, stride)) or [k0]
    out = []
    for s in starts:
        seqs = {tid: seq_from_track(d, s, window) for tid, d in tracks_k.items()
                if sum(s <= k < s + window for k in d) >= MIN_PRESENCE * window}
        for a, b in candidate_pairs(seqs):
            out.append((a, b, s, seqs[a], seqs[b]))
    return out


def build_pair_windows(tracks_k, window=WINDOW, stride=STRIDE, fps=FPS, starts=None):
    """-> list[(idA, idB, start_t, feats(window, 180))] (CLAUDE.md 7.1 mục 8)."""
    return [(a, b, s / fps, pair_features(A, B))
            for a, b, s, A, B in build_pair_instances(tracks_k, window, stride, starts)]


def clip_instances(npz, fps=FPS, window=WINDOW, stride=STRIDE):
    """Toàn bộ pipeline offline cho 1 clip: npz -> theo track -> lấy mẫu 15 FPS -> nối tracklet -> cặp/cửa sổ.
    Trả về (instances [(idA, idB, start_k, seqA, seqB)], tracks_k)."""
    tracks_k = link_tracklets(sample_tracks(tracks_from_npz(npz), fps, clip_times(npz)), fps)
    return build_pair_instances(tracks_k, window, stride), tracks_k


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
