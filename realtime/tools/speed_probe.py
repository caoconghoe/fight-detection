"""Đo tốc độ chạy (CHẠY NHANH, detectors/running.py) trên một video để hiệu chỉnh ngưỡng.

Chạy đúng pipeline realtime (YOLO-pose + ByteTrack mọi frame, lấy mẫu 15 FPS theo thời gian video, tham số
detectors.running trong config.yaml), KHÔNG hiển thị. Mỗi giây video in 1 dòng cho từng ID:
  L (px), tốc độ ngang / dọc / tổng (thân L/giây, trung vị trong giây đó, max trong ngoặc),
  dấu hiệu chân (gối = số lần góc gối đổi chiều, cc = khoảng cách 2 cổ chân đổi chiều, lp = nhịp luân phiên; lấy max),
  hướng, và lý do không đo được (thiếu hông/gối, ID mới, chạm mép dưới...).
Ghi thêm CSV từng tick vào realtime/events/speed_probe_<tên video>.csv.

Chạy (từ thư mục httm):  python realtime/tools/speed_probe.py <video> [--config ...] [--max-sec 120]
"""
import argparse
import contextlib
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.camera import Camera  # noqa: E402
from core.pose import PoseTracker  # noqa: E402
from detectors.running import RunningDetector  # noqa: E402
from shared.sampling import FpsSampler  # noqa: E402


def fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def summarize(rows, thr):
    """rows: dbg của 1 ID trong 1 giây -> 1 dòng chữ."""
    meas = [r for r in rows if "vt" in r]
    if not meas:
        whys = sorted({r.get("why", "?") for r in rows})
        return f"(khong do: {', '.join(whys)})"
    def med(key):
        v = [r[key] for r in meas if r.get(key) is not None]
        return (float(np.median(v)), float(np.max(v))) if v else (None, None)
    L, _ = med("L")
    (vh, vhx), (vd, vdx), (vt, vtx) = med("vh"), med("vd"), med("vt")
    knee = max(r["knee_sw"] for r in meas)
    ank = max(r["ank_sw"] for r in meas)
    alt = max(r["alt"] for r in meas)
    legs = sum(r["legs"] for r in meas) / len(meas)
    modes = defaultdict(int)
    for r in meas:
        modes[r["mode"]] += 1
    mode = max(modes, key=modes.get)
    stable = sum(r["stable"] for r in meas) / len(meas)
    flags = sorted({r.get("state") for r in meas if r.get("state")})
    mark = " >NGUONG" if vt is not None and vt > thr else ""
    return (f"L {fmt(L, 0):>4} | ngang {fmt(vh):>4} ({fmt(vhx)}) doc {fmt(vd):>4} ({fmt(vdx)}) "
            f"tong {fmt(vt):>4} ({fmt(vtx)}){mark} | chan goi{knee} cc{ank} lp{alt} ({legs:.0%} dat) "
            f"| {mode} on dinh {stable:.0%}" + (f" | {', '.join(flags)}" if flags else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--max-sec", type=float, default=0, help="chỉ đo N giây đầu (0 = cả video)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    pc = cfg["pose"]
    with contextlib.redirect_stdout(io.StringIO()):
        pose = PoseTracker(ROOT / pc["model"], pc["imgsz"], pc["conf"], pc["tracker"],
                           dedup_iou=pc["dedup_iou"], dedup_contain=pc["dedup_contain"])
    params = dict(cfg["detectors"].get("running") or {})
    params.pop("enabled", None)
    det = RunningDetector(fps=cfg["sample_fps"], full_body=cfg.get("full_body"), **params)
    cam = Camera(a.video, file_realtime=False)
    sampler = FpsSampler(cfg["sample_fps"])
    print(f"Video: {cam.name} | {cam.width}x{cam.height} | {cam.fps:.1f} FPS | nguong tong {det.speed_thr} than/s, "
          f"giu {det.min_sec} s | vfov {params.get('camera_vfov_deg', 50)} do")

    out = ROOT / "events" / f"speed_probe_{Path(a.video).stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fcsv = open(out, "w", newline="", encoding="utf-8-sig")
    w = csv.writer(fcsv)
    w.writerow(["t", "id", "L", "ngang", "doc", "tong", "goi", "co_chan", "luan_phien", "chan_dat", "huong",
                "on_dinh", "giu_s", "trang_thai"])
    sec, per_id, n_events = None, defaultdict(list), 0

    def flush(s):
        if s is None:
            return
        for tid in sorted(per_id):
            print(f"  {s:4d}-{s + 1:<4d}s ID {tid:<3} {summarize(per_id[tid], det.speed_thr)}")
        per_id.clear()

    while True:
        frame, t = cam.read()
        if frame is None or (a.max_sec and t >= a.max_sec):
            break
        tracks = pose(frame)
        if not sampler.tick(t):
            continue
        if sec is not None and int(t) != sec:
            flush(sec)
        sec = int(t)
        for e in det.update(frame, tracks, t):
            n_events += 1
            print(f"  >>> {t:6.2f} s  {e['type']} ID {e['ids']} | {e['detail']}")
        for tid in tracks:
            d = det.dbg.get(tid)
            if d is None:
                continue
            per_id[tid].append(d)
            if "vt" in d:
                w.writerow([f"{t:.3f}", tid, f"{d['L']:.1f}", f"{d['vh']:.2f}", fmt(d["vd"], 2), f"{d['vt']:.2f}",
                            d["knee_sw"], d["ank_sw"], d["alt"], int(d["legs"]), d["mode"], int(d["stable"]),
                            f"{d.get('held', 0):.2f}", d.get("state", "")])
            else:
                w.writerow([f"{t:.3f}", tid] + [""] * 11 + [d.get("why", "")])
    flush(sec)
    cam.release()
    fcsv.close()
    print(f"\nSo canh bao CHAY NHANH: {n_events} | CSV tung tick: {out}")


if __name__ == "__main__":
    main()
