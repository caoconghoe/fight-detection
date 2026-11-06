"""Hệ thống phát hiện hành vi bất thường – chương trình chính.

Chạy (từ thư mục httm):  python realtime/main.py [nguồn]
  nguồn: 0 (webcam) | http://<ip>:8080/video (điện thoại) | đường dẫn file video
Phím: q / Esc = thoát, s = lưu ảnh chụp màn hình, d = bật/tắt debug (bộ đếm từng luật).
"""
import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from core.camera import Camera, FpsMeter
from core.db import EventDB, slug
from core.events import EventManager, build_detectors
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
    pose = PoseTracker(ROOT / pc["model"], pc["imgsz"], pc["conf"], pc["tracker"],
                       dedup_iou=pc["dedup_iou"], dedup_contain=pc["dedup_contain"])
    sampler = FpsSampler(cfg["sample_fps"])
    manager = EventManager(build_detectors(cfg), **cfg["events"])
    print("Detector:", ", ".join(d.name for d in manager.detectors) or "(chua co)")
    uc = cfg["ui"]
    ui = UI(uc["font_size"], uc["kpt_conf"], uc["panel_width"], uc["new_sec"])
    win = uc["window"]
    snap_dir = ROOT / cfg["snapshot_dir"]
    print(f"Nguon: {cam.name} | backend: {cam.backend} | {cam.width}x{cam.height} | FPS khai bao: {cam.fps:.1f}")

    def save_snapshot(view, e):
        """Lưu ảnh (tên không dấu), trả về đường dẫn tuyệt đối hoặc None nếu lỗi."""
        snap_dir.mkdir(parents=True, exist_ok=True)
        ids = "-".join(str(i) for i in e["ids"])
        p = snap_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slug(e['type'])}_id{ids}.jpg"
        if cv2.imwrite(str(p), view):
            return p
        print("LOI luu anh:", p)
        return None

    if not a.no_show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    alerts = deque(maxlen=50)                     # cảnh báo gần đây, mới nhất trước
    highlight = set()
    WARMUP = 10                                   # bỏ 10 frame đầu khỏi thống kê (OpenVINO khởi động)
    n = n_sampled = 0
    pose_ms = None
    proc, samp = FpsMeter(), FpsMeter()           # số vòng lặp xử lý / số frame lấy mẫu mỗi giây
    t_first = t_now = start = None
    loop_t = time.perf_counter()
    status, status_t = [], 0.0
    debug_on = False                              # phím d: hiện bộ đếm từng luật trên khung mỗi người
    db = EventDB(ROOT / cfg["db_path"])
    session = db.start_session(source, cfg)
    print(f"Session #{session} | DB: {db.path}")
    reason = "loi / Ctrl+C"
    try:
        while True:
            frame, t = cam.read()
            if frame is None:
                reason = "het video / mat tin hieu"
                break

            t0 = time.perf_counter()
            tracks = pose(frame)
            ms = (time.perf_counter() - t0) * 1000
            pose_ms = ms if pose_ms is None else 0.9 * pose_ms + 0.1 * ms

            sampled = sampler.tick(t)
            events = []
            if sampled:                           # detector chỉ chạy ở frame lấy mẫu (15 FPS)
                events, highlight = manager.update(frame, tracks, t)
                for e in events:
                    e["time"], e["t"] = datetime.now().strftime("%H:%M:%S"), time.time()
                    alerts.appendleft(e)
                    print(f"[{e['time']}] {e['type']} | ID {e['ids']} | conf {e['confidence']}"
                          + (f" | {e['detail']}" if e.get("detail") else ""))

            proc.tick()
            if sampled:
                samp.tick()
            now = loop_t = time.perf_counter()
            n += 1
            if n == WARMUP:
                start, t_first = now, t
            elif n > WARMUP:
                t_now = t
                n_sampled += sampled

            if now - status_t >= 0.5:             # cập nhật chữ trạng thái 2 lần/giây (panel được cache)
                status_t = now
                status = [f"Camera: {cam.meter.value:.1f} FPS  |  Xử lý: {proc.value:.1f} FPS",
                          f"YOLO: {pose_ms:.0f} ms  |  Lấy mẫu: {samp.value:.1f} FPS",
                          f"Số người: {len(tracks)}",
                          f"Nguồn: {cam.name}"]
                if debug_on:
                    status += ["DEBUG (phím d để tắt)"] + manager.debug({})[1]
            dbg = manager.debug(tracks)[0] if debug_on else None
            view = ui.draw(frame, tracks, status, alerts, highlight, debug=dbg)
            for e in events:                      # ảnh chụp đã có khung đỏ + cảnh báo trên panel
                snap = save_snapshot(view, e)
                e["id"] = db.add_event(session, e["type"], e["severity"], e["ids"], e["confidence"], snap)

            if not a.no_show:
                cv2.imshow(win, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    reason = "phim q/Esc"
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    reason = "dong cua so (nut X)"
                    break
                if key == ord("d"):
                    debug_on, status_t = not debug_on, 0.0
                    print("Che do debug:", "BAT" if debug_on else "TAT")
                if key == ord("s"):
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    p = snap_dir / f"snap_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                    ok = cv2.imwrite(str(p), view)
                    print("Da luu anh:" if ok else "LOI luu anh:", p)
            if a.frames and n >= a.frames:
                reason = f"du {a.frames} frame"
                break
    finally:                                      # q, nút X, hết video, Ctrl+C hay lỗi đều đóng session
        db.end_session(session)
        db.close()
        cam.release()
        cv2.destroyAllWindows()
        print(f"Thoat: {reason} | da dong session #{session}")

    if t_now is not None:
        m = n - WARMUP
        print(f"So frame: {n} | Camera: {cam.meter.value:.1f} FPS | Xu ly trung binh: {m / (loop_t - start):.1f} FPS"
              f" | YOLO: {pose_ms:.1f} ms/frame")
        print(f"Lay mau: {n_sampled} frame trong {t_now - t_first:.1f} s -> {n_sampled / (t_now - t_first):.1f} FPS")


if __name__ == "__main__":
    main()
