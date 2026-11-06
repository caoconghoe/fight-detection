"""Hệ thống phát hiện hành vi bất thường – chương trình chính.

Chạy (từ thư mục httm):  python realtime/main.py [nguồn]
  nguồn: 0 (webcam) | http://<ip>:8080/video (điện thoại) | đường dẫn file video
Phím: q / Esc = thoát, s = lưu ảnh chụp màn hình.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from core.camera import Camera
from core.pose import PoseTracker
from core.ui import UI
from shared.sampling import FpsSampler

ROOT = Path(__file__).resolve().parent


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", help="0 | URL điện thoại | file video (mặc định lấy từ config)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-show", action="store_true", help="không mở cửa sổ (chạy thử / đánh giá)")
    ap.add_argument("--frames", type=int, default=0, help="dừng sau N frame (0 = không giới hạn)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    source = a.source if a.source is not None else cfg["source"]
    cam = Camera(source, **cfg["camera"])
    pc = cfg["pose"]
    pose = PoseTracker(ROOT / pc["model"], pc["imgsz"], pc["conf"], pc["tracker"])
    sampler = FpsSampler(cfg["sample_fps"])
    uc = cfg["ui"]
    ui = UI(uc["font_size"], uc["kpt_conf"], uc["panel_width"], uc["new_sec"])
    win = uc["window"]
    snap_dir = ROOT / cfg["snapshot_dir"]
    print(f"Nguon: {cam.name} | FPS nguon: {cam.fps:.1f}")

    if not a.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    alerts = []                                   # sẽ do EventManager cung cấp (bước A1)
    WARMUP = 10                                   # bỏ 10 frame đầu khỏi thống kê (OpenVINO khởi động)
    n = n_sampled = 0
    fps = pose_ms = None
    t_first = t_now = start = None
    loop_t = time.perf_counter()
    status, status_t = [], 0.0
    try:
        while True:
            frame, t = cam.read()
            if frame is None:
                break

            t0 = time.perf_counter()
            tracks = pose(frame)
            ms = (time.perf_counter() - t0) * 1000
            pose_ms = ms if pose_ms is None else 0.9 * pose_ms + 0.1 * ms

            sampled = sampler.tick(t)             # detector chạy khi sampled (bước A1+)

            now = time.perf_counter()
            dt, loop_t = now - loop_t, now
            if dt > 0:
                fps = 1 / dt if fps is None else 0.9 * fps + 0.1 / dt
            n += 1
            if n == WARMUP:
                start, t_first = now, t
            elif n > WARMUP:
                t_now = t
                n_sampled += sampled

            if now - status_t >= 0.5:             # cập nhật chữ trạng thái 2 lần/giây (panel được cache)
                status_t = now
                status = [f"FPS: {fps:.1f}  |  YOLO: {pose_ms:.0f} ms",
                          f"Số người: {len(tracks)}",
                          f"Nguồn: {cam.name}"]
            view = ui.draw(frame, tracks, status, alerts)

            if not a.no_show:
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("s"):
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    p = snap_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                    ok = cv2.imwrite(str(p), view)
                    print("Da luu anh:" if ok else "LOI luu anh:", p)
            if a.frames and n >= a.frames:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if t_now is not None:
        m = n - WARMUP
        print(f"So frame: {n} | FPS trung binh: {m / (loop_t - start):.1f} | YOLO: {pose_ms:.1f} ms/frame")
        print(f"Lay mau: {n_sampled} frame trong {t_now - t_first:.1f} s -> {n_sampled / (t_now - t_first):.1f} FPS")


if __name__ == "__main__":
    main()
