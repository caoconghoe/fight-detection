"""Hiệu chỉnh ngưỡng LUỒNG 2 (CLAUDE.md mục 12): chạy camera hoặc video BÌNH THƯỜNG 10–20 phút,
ghi điểm từng cửa sổ, in phân vị 95 / 99 / 99.9 và đề xuất ngưỡng. Không tự ghi vào config.

Chạy (từ thư mục httm):
  python realtime/tools/calibrate_scene.py 0 --minutes 15            # webcam
  python realtime/tools/calibrate_scene.py D:\\video\\binh_thuong.mp4   # video (chạy theo thời gian video)
Kết quả: realtime/events/calibration_scene_<thời gian>.csv (điểm từng cửa sổ).
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.camera import Camera  # noqa: E402
from detectors.scene import SceneScorer  # noqa: E402
from shared.sampling import FpsSampler  # noqa: E402
from shared import scene as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="0 = webcam, hoặc đường dẫn video bình thường")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    cam = Camera(a.source if not a.source.isdigit() else int(a.source), file_realtime=True)
    scorer = SceneScorer(threads=a.threads, backend="openvino")
    sampler = FpsSampler(S.FPS)
    rows, t_start, frames = [], time.time(), 0
    print(f"Ghi diem {a.minutes:.0f} phut tu '{a.source}' (chi dung khi canh BINH THUONG). Nhan Ctrl+C de dung som.")
    try:
        while time.time() - t_start < a.minutes * 60:
            frame, t = cam.read()
            if frame is None:
                break
            frames += 1
            if sampler.tick(t):
                scorer.submit(t, frame)
            for ts, s, g in scorer.drain():
                rows.append((round(ts, 3), round(s, 4), scorer.groups[int(np.argmax(g))]))
            if frames % 300 == 0:
                print(f"  {(time.time() - t_start) / 60:5.1f} phut, {len(rows)} cua so", flush=True)
    except KeyboardInterrupt:
        print("Dung som (Ctrl+C).")
    finally:
        time.sleep(1.0)
        for ts, s, g in scorer.drain():
            rows.append((round(ts, 3), round(s, 4), scorer.groups[int(np.argmax(g))]))
        scorer.close()
        cam.release()

    if len(rows) < 20:
        print(f"Qua it cua so ({len(rows)}) de tinh phan vi. Hay chay lau hon.")
        return
    scores = np.array([r[1] for r in rows])
    out = ROOT / "events" / f"calibration_scene_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "diem", "nhom_du_doan"])
        w.writerows(rows)

    p95, p99, p999 = np.percentile(scores, [95, 99, 99.9])
    print(f"\n{len(scores)} cua so, {(time.time() - t_start) / 60:.1f} phut, file: {out}")
    print(f"Phan vi  95  : {p95:+.3f}")
    print(f"Phan vi  99  : {p99:+.3f}")
    print(f"Phan vi 99.9 : {p999:+.3f}")
    print(f"Diem max     : {scores.max():+.3f}")
    print("DE XUAT: nguong = phan vi 99 (~1% cua so binh thuong vuot nguong). Phan vi 99.9 chi tin duoc khi co >= 1000 cua so.")
    print("Dat vao realtime/config.yaml -> detectors.scene.threshold. Khong tu dong ghi vao config.")


if __name__ == "__main__":
    main()
