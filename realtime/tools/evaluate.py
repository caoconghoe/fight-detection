"""Đánh giá hệ thống trên một thư mục video (CLAUDE.md D2) – chạy đúng pipeline realtime, KHÔNG hiển thị, không ghi DB.

Nhãn của mỗi video lấy từ TIỀN TỐ tên file (phần trước dấu "_" đầu tiên), xem LABELS bên dưới, vd:
  fight_abc.avi -> ĐÁNH NHAU · nonfight_xyz.avi -> không có sự kiện nào (mẫu âm) · fall_01.mp4 -> TÉ NGÃ
Tiền tố lạ -> vẫn chạy, in kết quả nhưng không tính vào precision/recall.

Đánh giá mức clip, với mỗi loại T:
  TP = clip nhãn T có báo T · FN = clip nhãn T không báo T · FP = clip nhãn khác (đã biết) có báo T
  "TÉ NGÃ RỒI NẰM BẤT ĐỘNG" được tính là đã báo cả TÉ NGÃ và NẰM BẤT ĐỘNG.
Xuất 2 file CSV (UTF-8, mở được bằng Excel) vào --out (mặc định kaggle/results/):
  eval_<thư mục>_<thời gian>_clips.csv   – mỗi video 1 dòng: cảnh báo, thời điểm, EMA đánh nhau lớn nhất, FPS...
  eval_<thư mục>_<thời gian>_metrics.csv – precision / recall / F1 / độ trễ TB theo loại

Chạy (từ thư mục httm):  python realtime/tools/evaluate.py data_selfrecorded/test [--config ...] [--out ...]
"""
import argparse
import contextlib
import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.camera import Camera  # noqa: E402
from core.events import PRIORITY, EventManager, build_detectors  # noqa: E402
from core.pose import PoseTracker  # noqa: E402
from shared.sampling import FpsSampler  # noqa: E402

FIGHT, FALL, STILL, FALL_STILL = "ĐÁNH NHAU", "TÉ NGÃ", "NẰM BẤT ĐỘNG", "TÉ NGÃ RỒI NẰM BẤT ĐỘNG"
NEGATIVE = None
LABELS = {                                   # tiền tố tên file -> loại sự kiện mong đợi (None = mẫu âm)
    "fight": FIGHT,
    "fall": FALL,
    "fallstill": FALL_STILL,
    "panic": "HOẢNG LOẠN",
    "run": "CHẠY NHANH", "running": "CHẠY NHANH",
    "handsup": "GIƠ TAY ĐẦU HÀNG",
    "wave": "VẪY TAY CẦU CỨU",
    "still": STILL, "lying": STILL,
    "distress": "ÔM ĐẦU / GỤC ĐẦU", "headhold": "ÔM ĐẦU / GỤC ĐẦU",
    "nonfight": NEGATIVE, "normal": NEGATIVE, "neg": NEGATIVE,
}
VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".webm"}


def label_of(path):
    prefix = path.stem.split("_")[0].lower()
    return (prefix, LABELS[prefix]) if prefix in LABELS else (prefix, "?")


def alerted_types(events):
    """Tập loại đã báo (TÉ NGÃ RỒI NẰM BẤT ĐỘNG tính là cả TÉ NGÃ và NẰM BẤT ĐỘNG)."""
    out = {e[0] for e in events}
    if FALL_STILL in out:
        out |= {FALL, STILL}
    return out


def first_time(events, typ):
    ts = [t for ty, t, _ in events if ty == typ or (ty == FALL_STILL and typ in (FALL, STILL))]
    return min(ts) if ts else None


def run_video(path, pose, cfg, quiet=True):
    cam = Camera(str(path), file_realtime=False)
    pose.reset()
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        mgr = EventManager(build_detectors(cfg), **cfg["events"])
    fight = next((d for d in mgr.detectors if d.name == "fight"), None)
    sampler = FpsSampler(cfg["sample_fps"])
    events, persons, yolo_ms, det_ms = [], [], [], []
    ema_max = p_max = 0.0
    n, t0, t_last = 0, time.perf_counter(), 0.0
    while True:
        frame, t = cam.read()
        if frame is None:
            break
        a = time.perf_counter()
        tracks = pose(frame)
        yolo_ms.append((time.perf_counter() - a) * 1000)
        persons.append(len(tracks))
        n, t_last = n + 1, t
        if sampler.tick(t):
            a = time.perf_counter()
            evs, _ = mgr.update(frame, tracks, t)
            det_ms.append((time.perf_counter() - a) * 1000)
            events += [(e["type"], t, e["ids"]) for e in evs]
            if fight is not None:
                ema_max = max([ema_max] + list(fight.ema.values()))
                p_max = max([p_max] + list(fight.last_p.values()))
    wall = time.perf_counter() - t0
    cam.release()
    return dict(events=events, n_frames=n, duration=t_last + 1 / cam.fps if n else 0.0, fps_src=cam.fps,
                fps_proc=n / wall if wall > 0 else 0.0, yolo_ms=float(np.mean(yolo_ms[10:] or yolo_ms or [0])),
                det_ms=float(np.mean(det_ms or [0])), persons_mean=float(np.mean(persons or [0])),
                persons_max=int(max(persons or [0])), fight_mode=getattr(fight, "mode", "-"),
                fight_ema_max=ema_max, fight_p_max=p_max)


def metrics(rows):
    """Precision / recall / F1 / độ trễ TB theo loại, mức clip (bỏ qua clip nhãn '?')."""
    known = [r for r in rows if r["expected"] != "?"]
    types = [t for t in PRIORITY if any(r["expected"] == t for r in known) or any(t in r["alerted"] for r in known)]
    out = []
    for t in types:
        pos = [r for r in known if r["expected"] == t]
        tp = [r for r in pos if t in r["alerted"]]
        fp = [r for r in known if r["expected"] != t and t in r["alerted"]]
        prec = len(tp) / (len(tp) + len(fp)) if tp or fp else None
        rec = len(tp) / len(pos) if pos else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec == 0 or rec == 0 else None)
        lat = [first_time(r["events"], t) for r in tp]
        out.append(dict(loai=t, so_clip_nhan=len(pos), TP=len(tp), FN=len(pos) - len(tp), FP=len(fp),
                        precision=prec, recall=rec, f1=f1, do_tre_TB_s=float(np.mean(lat)) if lat else None,
                        clip_bao_nham="; ".join(r["file"] for r in fp)))
    return out


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="thư mục video (nhãn theo tiền tố tên file)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT.parent / "kaggle" / "results"))
    a = ap.parse_args()

    folder = Path(a.folder)
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        print("Khong co video trong", folder)
        return
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    pc = cfg["pose"]
    pose = PoseTracker(ROOT / pc["model"], pc["imgsz"], pc["conf"], pc["tracker"],
                       dedup_iou=pc["dedup_iou"], dedup_contain=pc["dedup_contain"])
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        names = [d.name + (f" ({d.mode})" if hasattr(d, "mode") else "") for d in build_detectors(cfg)]
    print(buf.getvalue().strip())
    print(f"Danh gia {len(videos)} video trong {folder} | detector: {', '.join(names)}\n")

    rows = []
    for i, path in enumerate(videos, 1):
        prefix, expected = label_of(path)
        r = run_video(path, pose, cfg)
        alerted = alerted_types(r["events"])
        hit = None if expected == "?" else (expected in alerted if expected else not alerted)
        rows.append(dict(file=path.name, prefix=prefix, expected=expected, alerted=alerted, hit=hit, **r))
        exp_s = "?" if expected == "?" else (expected or "(khong co su kien)")
        alerts_s = ", ".join(f"{ty}@{t:.1f}s" for ty, t, _ in r["events"]) or "-"
        print(f"[{i}/{len(videos)}] {path.name}: nhan {exp_s} | bao: {alerts_s} | EMA danh nhau max "
              f"{r['fight_ema_max']:.2f} | {r['fps_proc']:.1f} FPS -> {'DUNG' if hit else '?' if hit is None else 'SAI'}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"eval_{folder.name}_{datetime.now():%Y%m%d_%H%M}"
    clip_csv, met_csv = out_dir / f"{stamp}_clips.csv", out_dir / f"{stamp}_metrics.csv"
    cols = ["file", "nhan", "loai_mong_doi", "dung", "canh_bao", "bao_dau_tien_s", "bao_dung_loai_s",
            "ema_danh_nhau_max", "p_danh_nhau_max", "che_do_danh_nhau", "thoi_luong_s", "fps_goc", "so_frame",
            "fps_xu_ly", "yolo_ms", "detector_ms", "so_nguoi_tb", "so_nguoi_max"]
    with open(clip_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            exp = r["expected"]
            w.writerow([r["file"], r["prefix"], "?" if exp == "?" else (exp or ""),
                        "" if r["hit"] is None else int(r["hit"]),
                        "; ".join(f"{ty} @{t:.2f}s ID{ids}" for ty, t, ids in r["events"]),
                        fmt(min((t for _, t, _ in r["events"]), default=None)),
                        fmt(first_time(r["events"], exp) if exp and exp != "?" else None),
                        fmt(r["fight_ema_max"], 3), fmt(r["fight_p_max"], 3), r["fight_mode"],
                        fmt(r["duration"]), fmt(r["fps_src"], 1), r["n_frames"], fmt(r["fps_proc"], 1),
                        fmt(r["yolo_ms"], 1), fmt(r["det_ms"], 2), fmt(r["persons_mean"], 1), r["persons_max"]])
    met = metrics(rows)
    with open(met_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(list(met[0]) if met else ["loai"])
        for m in met:
            w.writerow([fmt(v, 3) if isinstance(v, float) or v is None else v for v in m.values()])

    known = [r for r in rows if r["hit"] is not None]
    print(f"\nDung {sum(r['hit'] for r in known)}/{len(known)} clip co nhan "
          f"| FPS xu ly TB {np.mean([r['fps_proc'] for r in rows]):.1f} | YOLO TB {np.mean([r['yolo_ms'] for r in rows]):.1f} ms")
    print(f"\n{'Loai':26s} {'nhan':>5s} {'TP':>4s} {'FN':>4s} {'FP':>4s} {'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'tre TB':>7s}")
    for m in met:
        print(f"{m['loai']:26s} {m['so_clip_nhan']:5d} {m['TP']:4d} {m['FN']:4d} {m['FP']:4d} {fmt(m['precision']):>6s} "
              f"{fmt(m['recall']):>7s} {fmt(m['f1']):>6s} {fmt(m['do_tre_TB_s'], 1) + ' s' if m['do_tre_TB_s'] is not None else '-':>7s}")
    print(f"\nDa ghi:\n  {clip_csv}\n  {met_csv}")


if __name__ == "__main__":
    main()
