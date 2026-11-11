"""Chẩn đoán LUỒNG 2 trên một video (CLAUDE.md mục 12).

(a) Chấm điểm OFFLINE đúng cách notebook 04/05: lấy mẫu 2 FPS theo thời gian video (FpsSampler trên mọi frame),
    cửa sổ 16 frame, bước 1, chuẩn hoá mean/std rồi scene_head. In độ dài, số cửa sổ, min/TB/max, thời điểm max,
    nhóm gợi ý ở max và số giây liên tục dài nhất trên ngưỡng.
(b) --compare: chạy lại đường realtime (SceneDetector, nhịp 15 FPS như main.py) trên cùng video và so điểm từng cửa sổ.
(c) Nhiều video / thư mục: in thêm BẢNG tổng hợp (tên, train/test, độ dài, điểm max, giây liên tục trên ngưỡng,
    có báo không = có ≥ min_consecutive cửa sổ liên tiếp vượt ngưỡng như realtime) và ghi CSV vào kaggle/results/.
    Video UCF-Crime (tên kiểu Robbery001_x264) được xếp train/test nếu có --split-list (Anomaly_Test.txt hoặc
    file chú thích thời gian Temporal_Anomaly_Annotation*.txt): có tên trong danh sách = test, không có = train;
    không có danh sách -> "không rõ".
Ngưỡng / min_consecutive mặc định lấy từ config.yaml (detectors.scene).

Chạy (từ thư mục httm):
  python realtime/tools/scene_score_video.py <video>
  python realtime/tools/scene_score_video.py <video> --compare --threshold -0.8
  python realtime/tools/scene_score_video.py data_selfrecorded/test [--split-list Anomaly_Test.txt]
"""
import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detectors.scene import SceneDetector, SceneScorer  # noqa: E402
from shared import scene as S  # noqa: E402
from shared.sampling import FpsSampler  # noqa: E402


def read_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if 1 <= fps <= 120 else 30.0


def offline(path, scorer, max_sec=S.MAX_SEC):
    """-> dict: dur (s), n_frames (2 FPS), starts/ends (s) của từng cửa sổ, scores, groups."""
    cap = cv2.VideoCapture(str(path))
    fps = read_fps(cap)
    smp, feats, ts, i = FpsSampler(S.FPS), [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t >= max_sec:
            break
        if smp.tick(t):
            v = np.asarray(scorer.backbone(S.prep(f)[None]), np.float32).reshape(-1)
            feats.append((v - scorer.mean) / scorer.std)
            ts.append(t)
    cap.release()
    dur = i / fps
    win = scorer.win
    n = len(feats)
    if n < win:
        return dict(dur=dur, n_frames=n, starts=np.array([]), ends=np.array([]), scores=np.array([]), groups=[])
    V = np.stack(feats).astype(np.float32)
    X = np.stack([V[k:k + win] for k in range(n - win + 1)])
    sc, gl = [], []
    for c in range(0, len(X), 256):
        s, g = scorer.head.run(None, {scorer.head_in: X[c:c + 256]})
        sc.append(np.ravel(s))
        gl.append(g)
    sc, gl = np.concatenate(sc), np.concatenate(gl)
    ts = np.array(ts)
    starts = ts[: n - win + 1]
    ends = ts[win - 1:]
    return dict(dur=dur, n_frames=n, starts=starts, ends=ends, scores=sc,
                groups=[scorer.groups[int(np.argmax(g))] for g in gl])


def longest_run_seconds(starts, ends, scores, thr):
    best, cur_s, cur_e, run_start = 0.0, None, None, None
    for a, b, s in zip(starts, ends, scores):
        if s > thr:
            if run_start is None:
                run_start = a
            cur_e = b
            best = max(best, cur_e - run_start)
        else:
            run_start = None
    return best


def report(res, thr):
    print(f"Do dai video     : {res['dur']:.1f} s  (lay toi da {S.MAX_SEC} s)")
    print(f"So khung 2 FPS   : {res['n_frames']}")
    if res["scores"].size == 0:
        print(f"!! CANH BAO: chi co {res['n_frames']} khung 2 FPS, can 16 khung (~8 s) -> KHONG co cua so nao, luong 2 khong the bao.")
        return
    sc = res["scores"]
    k = int(np.argmax(sc))
    print(f"So cua so        : {len(sc)} (buoc 1 = 0.5 s)")
    print(f"Diem min / TB / max: {sc.min():+.3f} / {sc.mean():+.3f} / {sc.max():+.3f}")
    print(f"Max tai t = {res['ends'][k]:.1f} s (cua so {res['starts'][k]:.1f}-{res['ends'][k]:.1f} s), nhom gợi ý: {res['groups'][k]}")
    run = longest_run_seconds(res["starts"], res["ends"], sc, thr)
    print(f"Chuoi lien tuc tren nguong {thr}: dai nhat {run:.1f} s")
    print(f"Cua so tren nguong: {int((sc > thr).sum())} / {len(sc)}")


def realtime_windows(path, threshold):
    """Đường realtime: SceneDetector nhận frame ở nhịp 15 FPS (như main.py), trả (t_end, điểm)."""
    det = SceneDetector(threshold=threshold)
    if det.scorer is None:
        raise SystemExit("Khong co mo hinh luong 2")
    cap = cv2.VideoCapture(str(path))
    fps = read_fps(cap)
    smp15, i, t_last = FpsSampler(15), 0, 0.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        t_last = t
        if smp15.tick(t):
            det.update(f, {}, t)
    cap.release()
    for _ in range(100):                                   # chờ luồng nền xử lý hết hàng đợi
        if det.scorer.q.empty():
            break
        time.sleep(0.1)
    time.sleep(1.0)
    det.update(None, {}, t_last)                           # kéo nốt kết quả
    hist = list(det.history)
    det.close()
    return hist


def compare(path, off, threshold):
    hist = realtime_windows(path, threshold)
    if not hist:
        print("Realtime: khong co cua so nao.")
        return
    rt_t = np.array([h[0] for h in hist])
    rt_s = np.array([h[1] for h in hist])
    off_t, off_s = off["ends"], off["scores"]
    diffs, dt = [], []
    for t, s in zip(rt_t, rt_s):
        j = int(np.argmin(np.abs(off_t - t)))
        if abs(off_t[j] - t) < 0.5:
            diffs.append(abs(s - off_s[j]))
            dt.append(t - off_t[j])
    print("\n== SO SANH realtime va offline")
    print(f"Cua so realtime: {len(rt_s)} | offline: {len(off_s)} | ghep duoc: {len(diffs)}")
    if diffs:
        print(f"Do lech diem: max {max(diffs):.4f} | TB {np.mean(diffs):.4f}")
        print(f"Lech thoi gian moc cua so: TB {np.mean(dt):+.3f} s (max {np.max(np.abs(dt)):.3f} s)")
    print(f"Diem realtime max {rt_s.max():+.3f} | offline max {off_s.max():+.3f}")


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".webm"}
UCF_NAME = re.compile(r"^([A-Za-z]+?)(\d{3})(_x264)?$")


def ucf_key(stem):
    """'Robbery001_x264' -> 'robbery001' (None nếu không giống tên UCF-Crime)."""
    m = UCF_NAME.match(stem)
    return f"{m.group(1).lower()}{m.group(2)}" if m else None


def load_split_list(path):
    """Tập khoá UCF trong Anomaly_Test.txt / file chú thích thời gian (mỗi dòng chứa tên file video)."""
    keys = set()
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        for tok in re.split(r"[\s/\\]+", line.strip()):
            k = ucf_key(Path(tok).stem) if tok else None
            if k:
                keys.add(k)
    return keys


def split_of(stem, test_keys):
    k = ucf_key(stem)
    if k is None:
        return "-"                                   # không phải video UCF-Crime
    if test_keys is None:
        return "không rõ"
    return "test" if k in test_keys else "train"


def max_consecutive(scores, thr):
    best = cur = 0
    for s in scores:
        cur = cur + 1 if s > thr else 0
        best = max(best, cur)
    return best


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    sc_cfg = cfg["detectors"]["scene"]
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="file video hoặc thư mục")
    ap.add_argument("--threshold", type=float, default=float(sc_cfg["threshold"]))
    ap.add_argument("--min-consecutive", type=int, default=int(sc_cfg.get("min_consecutive", 2)))
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--split-list", nargs="*", default=[], help="Anomaly_Test.txt / Temporal_Anomaly_Annotation*.txt")
    a = ap.parse_args()
    paths = []
    for v in a.videos:
        p = Path(v)
        paths += sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
    test_keys = None
    if a.split_list:
        test_keys = set().union(*(load_split_list(f) for f in a.split_list))
        print(f"Danh sach test: {len(test_keys)} video UCF tu {', '.join(a.split_list)}")
    scorer = SceneScorer(backend="openvino", threads=2)
    rows = []
    for path in paths:
        t0 = time.perf_counter()
        res = offline(path, scorer)
        print(f"\n== OFFLINE: {path.name}  ({time.perf_counter() - t0:.0f} s xu ly)")
        report(res, a.threshold)
        sc = res["scores"]
        n_cons = max_consecutive(sc, a.threshold) if sc.size else 0
        rows.append(dict(ten=path.name, tap=split_of(path.stem, test_keys), do_dai_s=round(res["dur"], 1),
                         so_cua_so=int(sc.size), diem_max=round(float(sc.max()), 3) if sc.size else None,
                         nhom_o_max=res["groups"][int(np.argmax(sc))] if sc.size else "",
                         giay_lien_tuc=round(longest_run_seconds(res["starts"], res["ends"], sc, a.threshold), 1)
                         if sc.size else 0.0,
                         cua_so_lien_tiep=n_cons,
                         bao="CO" if n_cons >= a.min_consecutive else ("KHONG (<16 khung)" if not sc.size else "KHONG")))
        if a.compare:
            compare(path, res, a.threshold)
    scorer.close()
    if len(rows) > 1:
        print(f"\n== BANG (nguong {a.threshold}, bao khi >= {a.min_consecutive} cua so lien tiep)")
        print(f"{'ten':42s} {'tap':9s} {'dai(s)':>7s} {'max':>8s} {'giay>ng':>8s} {'cs lt':>6s}  bao  nhom o max")
        for r in rows:
            mx = "-" if r["diem_max"] is None else f"{r['diem_max']:+.3f}"
            print(f"{r['ten'][:42]:42s} {r['tap']:9s} {r['do_dai_s']:7.1f} {mx:>8s} {r['giay_lien_tuc']:8.1f} "
                  f"{r['cua_so_lien_tiep']:6d}  {r['bao']}  {r['nhom_o_max']}")
        out = ROOT.parent / "kaggle" / "results" / f"scene_score_{datetime.now():%Y%m%d_%H%M}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {out}")


if __name__ == "__main__":
    main()
