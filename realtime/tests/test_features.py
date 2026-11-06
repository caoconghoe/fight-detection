"""Unit test shared/features.py bằng dữ liệu giả.  Chạy (từ thư mục httm):  python realtime/tests/test_features.py"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import features as F  # noqa: E402
from synth import person  # noqa: E402

W_IMG, H_IMG = 1280, 720


def scene(t, fight=True, gap=0.0):
    """2 người đứng gần nhau (ID 1, 2); fight: tay phải của A đấm qua lại 2 Hz về phía B."""
    rng = np.random.default_rng(int(t * 1000))
    rw = (-0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * 2 * t)), 0.05) if fight else None
    return {1: person(560, 450, H=300, rw=rw, noise=0.5, rng=rng),
            2: person(760 + gap, 450, H=300, noise=0.5, rng=rng)}


def fake_npz(fps, secs=5.0, fight=True, gap=0.0, drop=None):
    """npz giống đầu ra 01_extract_pose: data (N, 57) toạ độ chuẩn hoá 0..1 theo W, H riêng."""
    n = int(round(secs * fps))
    rows = []
    for i in range(n):
        for tid, tr in scene(i / fps, fight, gap).items():
            if drop and drop(i, tid):
                continue
            box = tr["box"] / [W_IMG, H_IMG, W_IMG, H_IMG]
            k = tr["kpts"].copy()
            k[:, 0] /= W_IMG
            k[:, 1] /= H_IMG
            rows.append(np.concatenate([[i, tid], box, k.ravel()]))
    return {"data": np.array(rows, np.float32), "fps": fps, "n_frames": n, "width": W_IMG, "height": H_IMG}


def test_tracks_from_npz_pixels():
    npz = fake_npz(30, 1.0)
    tr = F.tracks_from_npz(npz)
    t, box, k = tr[1][3]
    ref = scene(3 / 30)[1]
    assert abs(t - 0.1) < 1e-6
    assert np.allclose(box, ref["box"], atol=0.05) and np.allclose(k[:, :2], ref["kpts"][:, :2], atol=0.05)


def test_track_minus_one_dropped():
    npz = fake_npz(30, 1.0)
    npz["data"][::2, 1] = -1
    assert -1 not in F.tracks_from_npz(npz)


def test_sampling_15fps_grid():
    for fps in (30, 25, 22, 29.6, 24.9, 60):
        npz = fake_npz(fps, 5.0)
        tk = F.sample_tracks(F.tracks_from_npz(npz), 15, F.clip_times(npz))
        ks = sorted(tk[1])
        assert len(ks) in (75, 76), (fps, len(ks))                   # 5 s × 15 FPS
        assert ks == list(range(ks[0], ks[0] + len(ks))), fps         # liên tục, không trùng, không hở


def test_fps_invariance():
    """Cùng chuyển động quay ở 30/25/22 FPS -> cùng số cửa sổ, đặc trưng gần như nhau (vận tốc cùng thang)."""
    res = {}
    for fps in (30, 25, 22):
        inst, _ = F.clip_instances(fake_npz(fps, 5.0))
        feats = np.stack([F.pair_features(A, B) for _, _, _, A, B in inst])
        res[fps] = feats
    assert len({len(v) for v in res.values()}) == 1, {k: len(v) for k, v in res.items()}
    vel = {f: np.abs(v[..., F.SL_VEL]).mean() for f, v in res.items()}
    for f in (25, 22):
        assert abs(vel[f] / vel[30] - 1) < 0.1, vel
        assert np.abs(res[f] - res[30]).mean() < 0.02, (f, np.abs(res[f] - res[30]).mean())


def test_link_tracklets():
    def split(gap_frames, shift=0.0):
        """ID 1 mất từ frame 60 rồi xuất hiện lại với ID 7 sau gap_frames frame (30 FPS)."""
        npz = fake_npz(30, 5.0, drop=lambda i, tid: tid == 1 and 60 <= i < 60 + gap_frames)
        d = npz["data"]
        later = (d[:, 1] == 1) & (d[:, 0] >= 60 + gap_frames)
        d[later, 1] = 7
        d[later, 2:6:2] += shift
        return F.link_tracklets(F.sample_tracks(F.tracks_from_npz(npz), 15, F.clip_times(npz)))
    assert 7 not in split(9) and len(split(9)[1]) >= 70                # vắng 0.3 s -> gộp
    assert 7 not in split(24)                                          # vắng 0.8 s -> gộp (link_gap_sec 1.0)
    assert 7 in split(45)                                              # vắng 1.5 s -> không gộp
    assert 7 in split(24, shift=0.3)                                   # nhảy 384 px > 1.0 × 300 -> không gộp
    assert 7 in split(9, shift=0.3)                                    # nhảy xa -> không gộp


def test_fill_gaps():
    box = np.zeros((10, 4), np.float32)
    kp = np.zeros((10, 17, 3), np.float32)
    present = np.zeros(10, bool)
    for i in (0, 3, 9):                                                # khoảng trống 2 frame và 5 frame
        box[i] = [i, i, i + 10, i + 20]
        kp[i, :, :2], kp[i, :, 2] = i, 0.9
        present[i] = True
    valid = F.fill_gaps(box.copy(), kp.copy(), present, max_fill=3)
    assert valid.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    assert F.fill_gaps(box.copy(), kp.copy(), present).all()            # mặc định lấp tới 8 frame
    valid = F.fill_gaps(box, kp, present, max_fill=3)
    assert np.allclose(kp[1, :, 0], 1) and np.allclose(box[2, 0], 2)
    assert np.all(kp[5] == 0)


def test_feature_shape_and_pairs():
    wins = F.build_pair_windows(F.clip_instances(fake_npz(30))[1])
    assert len(wins) == (75 - 32) // 8 + 1 + 1                         # 6 cửa sổ trượt + 1 cửa sổ cuối
    a, b, t0, f = wins[1]
    assert (a, b) == (1, 2) and abs(t0 - 8 / 15) < 1e-6 and f.shape == (32, F.FEAT_DIM) and f.dtype == np.float32
    assert np.isfinite(f).all()


def test_far_people_no_pair():
    assert F.clip_instances(fake_npz(30, gap=500))[0] == []           # cách 700 px > 2.0 × 300
    assert F.clip_instances(fake_npz(30, gap=350))[0] != []           # cách 550 px ≤ 2.0 × 300 (mức cũ 1.5 thì không)
    assert F.clip_instances(fake_npz(30, gap=350), near_ratio=1.5)[0] == []


def test_single_person_no_pair():
    npz = fake_npz(30, drop=lambda i, tid: tid == 2)
    assert F.clip_instances(npz)[0] == []


def test_max_pairs():
    rng = np.random.default_rng(0)
    tk = {tid: {k: person(400 + 60 * tid, 450, H=300, rng=rng) for k in range(32)} for tid in range(8)}
    tk = {tid: {k: (v["box"], v["kpts"]) for k, v in d.items()} for tid, d in tk.items()}
    assert len(F.build_pair_instances(tk)) == F.PARAMS["max_pairs"]


def test_low_conf_zeroed():
    inst, _ = F.clip_instances(fake_npz(30))
    _, _, _, A, B = inst[0]
    A = {k: v.copy() for k, v in A.items()}
    A["kpts"][:, 9, 2] = 0.2                                           # cổ tay trái A conf thấp
    f = F.pair_features(A, B)
    assert np.all(f[:, 18:20] == 0) and np.all(f[:, 68 + 9] == 0) and np.all(f[:, 102 + 18:102 + 20] == 0)
    assert np.all(f[:, 170:172] == F.FAR)                              # khoảng cách từ cổ tay thiếu = FAR


def test_translation_scale_invariance():
    _, _, _, A, B = F.clip_instances(fake_npz(30))[0][0]
    f = F.pair_features(A, B)
    def tf(s, scale, shift):
        s = {k: v.copy() for k, v in s.items()}
        s["kpts"][..., :2] = s["kpts"][..., :2] * scale + shift
        s["box"] = s["box"] * scale + np.tile(shift, 2)
        return s
    g = F.pair_features(tf(A, 2.0, [100, -50]), tf(B, 2.0, [100, -50]))
    assert np.allclose(f, g, atol=1e-4)


def test_swap_and_flip_consistent():
    _, _, _, A, B = F.clip_instances(fake_npz(30))[0][0]
    f = F.pair_features(A, B)
    assert np.allclose(F.pair_features(B, A), F.swap_ab(f), atol=1e-5)
    def flip(s):
        s = {k: v.copy() for k, v in s.items()}
        s["kpts"] = s["kpts"][:, F.FLIP_KPTS]
        s["kpts"][..., 0] *= -1
        s["box"] = np.stack([-s["box"][:, 2], s["box"][:, 1], -s["box"][:, 0], s["box"][:, 3]], -1)
        return s
    assert np.allclose(F.pair_features(flip(A), flip(B)), F.flip_h(f), atol=1e-5)


def test_batch_equals_single():
    inst, _ = F.clip_instances(fake_npz(30))
    As = {k: np.stack([i[3][k] for i in inst]) for k in inst[0][3]}
    Bs = {k: np.stack([i[4][k] for i in inst]) for k in inst[0][4]}
    single = np.stack([F.pair_features(i[3], i[4]) for i in inst])
    assert np.allclose(F.pair_features(As, Bs), single, atol=1e-6)


def test_fight_moves_more_than_still():
    """Vận tốc cổ tay phải của A (khớp 10 -> cột 102 + 20, 21) khi đấm >> khi đứng yên."""
    v = lambda fight: np.abs(np.stack([F.pair_features(A, B) for *_, A, B in F.clip_instances(  # noqa: E731
        fake_npz(30, fight=fight))[0]])[..., 122:124]).mean()
    assert v(True) > 5 * v(False), (v(True), v(False))


def test_short_clip_padded():
    npz = fake_npz(30, 1.2)                                            # 18 frame mẫu < 32
    inst, _ = F.clip_instances(npz)
    assert len(inst) == 1 and inst[0][3]["valid"].sum() == 18 and np.all(inst[0][3]["kpts"][18:] == 0)
    assert F.clip_instances(npz, pad_short=False)[0] == []


def test_tail_window_covers_end():
    assert F.window_starts(0, 74) == [0, 8, 16, 24, 32, 40, 43]
    assert F.window_starts(0, 71) == [0, 8, 16, 24, 32, 40]             # vừa khít -> không thêm
    assert F.window_starts(0, 74, tail_window=False)[-1] == 40


def test_unassigned_detections_attached():
    npz = fake_npz(30, 3.0)
    d = npz["data"]
    d[(d[:, 1] == 2) & (d[:, 0] % 3 == 0), 1] = -1                    # 1/3 detection của ID 2 chưa có ID
    n_full = len(F.tracks_from_npz(fake_npz(30, 3.0))[2])
    assert len(F.tracks_from_npz(npz)[2]) == n_full                     # gán lại đủ
    assert len(F.tracks_from_npz(npz, assign_iou=None)[2]) < n_full
    assert sum(len(v) for v in F.tracks_from_npz(npz).values()) == len(d)   # không gán nhầm sang ID 1


def test_copresence_40():
    # ID 2 chỉ xuất hiện ở 14/32 frame đầu (44%) -> đạt 40%, không đạt 60%
    npz = fake_npz(30, 32 / 15, drop=lambda i, tid: tid == 2 and i >= 28)
    assert F.clip_instances(npz)[0] != []
    assert F.clip_instances(npz, min_copresence=0.6)[0] == []


def test_explain_reasons():
    def why(npz):
        tk = F.clip_tracks(npz)
        n = len(F.sample_indices(list(F.clip_times(npz)), F.FPS))
        return F.explain_no_pairs(tk, n)
    assert why(fake_npz(30)) is None
    assert why(fake_npz(30, 0.6)) is None                               # clip ngắn đủ 2 người: đệm -> có instance
    assert why(fake_npz(30, 0.6, drop=lambda i, tid: tid == 2)) == "a_clip_ngan"
    assert why(fake_npz(30, drop=lambda i, tid: tid == 2)) == "b_khong_co_2_nguoi"
    assert why(fake_npz(30, drop=lambda i, tid: (tid == 2) == (i < 75))) == "b_khong_co_2_nguoi"   # thay phiên
    assert why(fake_npz(30, drop=lambda i, tid: tid == 2 and (i // 10) % 8 != 0)) == "c_khong_du_co_mat_chung"  # ≤ 16% < 30%
    assert why(fake_npz(30, gap=500)) == "d_qua_xa"


def test_unknown_param_rejected():
    try:
        F.clip_instances(fake_npz(30, 1.0), min_presence=0.5)
    except TypeError:
        return
    raise AssertionError("tham so la phai bao loi")


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
