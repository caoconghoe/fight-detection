"""Duyệt các sự kiện chưa review (CLAUDE.md A6).

Mỗi sự kiện: phát lặp clip (nếu có) hoặc ảnh chụp, kèm thông tin. Phím: y = đúng, n = báo nhầm,
s = bỏ qua, q / Esc / nút X = thoát. Cuối cùng in bảng thống kê db.stats() (precision theo loại).
Chạy (từ thư mục httm):  python realtime/tools/review.py [--db realtime/events/events.db] [--stats]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.db import EventDB  # noqa: E402
from core.ui import fit, load_font  # noqa: E402

WIN = "HTTM - Duyet su kien (y: dung, n: bao nham, s: bo qua, q: thoat)"
MAX_W = 1280


def resolve(p):
    if not p:
        return None
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    return p if p.is_file() else None


def header(width, lines, size=20):
    """Thanh thông tin (chữ có dấu, Pillow): dòng đầu vàng đậm, dòng cuối là hướng dẫn phím."""
    font, bold = load_font(size), load_font(size, bold=True)
    lh, pad = int(size * 1.45), 10
    img = Image.new("RGB", (width, pad * 2 + lh * len(lines)), (28, 28, 32))
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        f = bold if i == 0 else font
        color = (255, 210, 0) if i == 0 else (120, 220, 255) if i == len(lines) - 1 else (235, 235, 235)
        d.text((pad, pad + i * lh), fit(f, s, width - 2 * pad), font=f, fill=color)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def fit_width(img, w):
    h0, w0 = img.shape[:2]
    return img if w0 == w else cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)


def blank(text):
    img = np.full((360, 640, 3), 40, np.uint8)
    cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return img


def review_loop(db, events):
    """-> (số đúng, số báo nhầm, số bỏ qua, đã thoát giữa chừng?)"""
    n_true = n_false = n_skip = 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for i, ev in enumerate(events, 1):
        clip, snap = resolve(ev["clip_path"]), resolve(ev["snapshot_path"])
        cap = cv2.VideoCapture(str(clip)) if clip else None
        still = cv2.imread(str(snap)) if snap else None
        if still is None:
            still = blank("Khong co anh / clip")
        fps = cap.get(cv2.CAP_PROP_FPS) if cap else 0
        delay = int(1000 / fps) if fps and fps > 0 else 50
        ids = ", ".join(str(x) for x in json.loads(ev["track_ids"] or "[]"))
        conf = f"  ·  độ tin cậy {ev['confidence']:.2f}" if ev["confidence"] is not None else ""
        width = min(MAX_W, still.shape[1])
        head = header(width, [
            f"#{ev['id']}  {ev['type']}   ({i}/{len(events)})",
            f"{ev['ts']}  ·  ID {ids}{conf}  ·  {'clip' if cap else 'ảnh chụp' if snap else 'không có ảnh'}",
            "y: ĐÚNG     n: BÁO NHẦM     s: bỏ qua     q: thoát"])
        while True:
            frame = still
            if cap:
                ok, f = cap.read()
                if not ok:                                         # hết clip -> phát lại
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                if ok:
                    frame = f
            cv2.imshow(WIN, np.vstack([head, fit_width(frame, width)]))
            key = cv2.waitKey(delay) & 0xFF
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), 27):
                if cap:
                    cap.release()
                cv2.destroyAllWindows()
                return n_true, n_false, n_skip, True
            if key in (ord("y"), ord("n")):
                db.review(ev["id"], key == ord("y"))
                n_true, n_false = n_true + (key == ord("y")), n_false + (key == ord("n"))
                print(f"#{ev['id']} {ev['type']}: {'DUNG' if key == ord('y') else 'BAO NHAM'}")
                break
            if key == ord("s"):
                n_skip += 1
                break
        if cap:
            cap.release()
    cv2.destroyAllWindows()
    return n_true, n_false, n_skip, False


def print_stats(db):
    rows = db.stats()
    print(f"\n{'Loai':26s} {'Tong':>5s} {'Da duyet':>9s} {'Dung':>5s} {'Precision':>10s}")
    print("-" * 59)
    for r in rows:
        prec = "-" if r["precision"] is None else f"{r['precision']:.2f}"
        print(f"{r['type']:26s} {r['total']:5d} {r['reviewed'] or 0:9d} {r['correct'] or 0:5d} {prec:>10s}")
    if not rows:
        print("(chua co su kien)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "events" / "events.db"))
    ap.add_argument("--stats", action="store_true", help="chỉ in bảng thống kê, không mở cửa sổ duyệt")
    a = ap.parse_args()
    if not Path(a.db).is_file():
        print("Khong thay DB:", a.db)
        return
    db = EventDB(a.db)
    try:
        if not a.stats:
            events = db.unreviewed()
            print(f"{len(events)} su kien chua duyet")
            if events:
                t, f, s, quit_early = review_loop(db, events)
                print(f"Da duyet: {t} dung, {f} bao nham, {s} bo qua" + (" (thoat giua chung)" if quit_early else ""))
        print_stats(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
