"""Giao diện cửa sổ giám sát: video (khung người, khung xương, băng cảnh báo) + panel bên phải.

- Trên video vẽ bằng cv2 (nhanh): khung góc, khung xương 2 màu trái/phải, nhãn ID, viền đỏ nhấp nháy khi có
  cảnh báo. Băng cảnh báo trên đầu video có chữ tiếng Việt -> vẽ bằng Pillow MỘT lần cho mỗi nội dung rồi trộn vào.
- Panel (Pillow, vì cv2.putText không có dấu): tiêu đề + đồng hồ + trạng thái, 4 ô số liệu, thanh điểm luồng 2,
  danh sách cảnh báo dạng thẻ (màu theo mức ưu tiên), dòng phím tắt. Panel chỉ vẽ lại khi nội dung đổi.
"""
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}                      # khớp bên trái (chỉ số lẻ, COCO-17)
FONTS = [("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
         ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
         ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
         ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]

# ---- màu trên video (BGR) ----
C_OK, C_ALERT, C_FIGHT = (110, 200, 90), (0, 150, 255), (60, 60, 240)   # bình thường / trong cảnh báo / đánh nhau
C_LEFT, C_RIGHT, C_MID = (240, 190, 70), (70, 160, 250), (225, 225, 225)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)

# ---- màu panel (RGB) ----
P_BG, P_HEAD_BG, P_CARD, P_CARD_NEW = (22, 24, 30), (32, 35, 44), (36, 39, 49), (58, 32, 38)
P_TEXT, P_DIM, P_FAINT, P_SEP = (236, 238, 242), (150, 156, 170), (95, 100, 115), (55, 59, 72)
P_GREEN, P_RED, P_ACCENT = (52, 199, 120), (239, 68, 68), (96, 165, 250)


def sev_color(sev):
    """Màu theo mức ưu tiên (RGB): đỏ ≥ 90, cam ≥ 70, vàng ≥ 55, tím = cảnh bất thường (50), xanh còn lại."""
    sev = sev or 0
    if sev >= 90:
        return (239, 68, 68)
    if sev >= 70:
        return (249, 115, 22)
    if sev >= 55:
        return (234, 179, 8)
    if sev >= 50:
        return (168, 85, 247)
    return (59, 130, 246)


def load_font(size, bold=False):
    for reg, bd in FONTS:
        f = bd if bold else reg
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def fit(font, s, width):
    """Cắt chuỗi cho vừa chiều rộng (thêm '…')."""
    if font.getlength(s) <= width:
        return s
    while s and font.getlength(s + "…") > width:
        s = s[:-1]
    return s + "…"


def _fmt(v, nd=1, unit=""):
    return "–" if v is None else f"{v:.{nd}f}{unit}"


class UI:
    def __init__(self, font_size=18, kpt_conf=0.3, panel_width=360, new_sec=10.0, alert_sec=5.0, min_height=540):
        self.min_height = min_height                        # video thấp hơn -> phóng lên trước khi vẽ (panel đủ chỗ, chữ nét)
        self.font_size, self.panel_width = font_size, panel_width
        self.kpt_conf, self.new_sec, self.alert_sec = kpt_conf, new_sec, alert_sec
        self.fonts = {}
        self.cache_key, self.cache_img = None, None         # panel chỉ vẽ lại khi nội dung đổi
        self.banner_key, self.banner_img = None, None       # băng cảnh báo trên video

    def _fonts(self, size):
        size = max(8, int(size))
        if size not in self.fonts:
            self.fonts[size] = (load_font(size), load_font(size, bold=True))
        return self.fonts[size]

    # ================= trên video =================
    def draw_debug(self, img, box, lines, top):
        """Chữ debug (ASCII, cv2 cho nhanh) ở góc trên trái khung người, ngay dưới nhãn ID."""
        H, W = img.shape[:2]
        x = min(max(int(box[0]), 0) + 2, W - 10)
        y = top
        for s in lines:
            (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            if y + th + 6 > H:
                break
            cv2.rectangle(img, (x, y), (x + tw + 6, y + th + 6), BLACK, -1)
            cv2.putText(img, s, (x + 3, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            y += th + 6

    @staticmethod
    def _corner_box(img, x1, y1, x2, y2, color, t):
        """Khung mảnh + 4 góc đậm."""
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        L = max(8, int(0.22 * min(x2 - x1, y2 - y1)))
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)

    def draw_tracks(self, img, tracks, highlight=(), debug=None, alert_ids=()):
        H, W = img.shape[:2]
        s = max(1.0, H / 540)
        lw = max(2, int(2 * s))
        fscale, fth = 0.55 * s, max(1, int(1.5 * s))
        for tid, tr in tracks.items():
            if tid in highlight:
                color = C_FIGHT
            elif tid in alert_ids:
                color = C_ALERT
            else:
                color = C_OK
            x1, y1, x2, y2 = tr["box"].astype(int)
            self._corner_box(img, x1, y1, x2, y2, color, lw + (2 if tid in highlight else 1))
            k = tr["kpts"]
            vis = k[:, 2] >= self.kpt_conf
            for a, b in SKELETON:
                if vis[a] and vis[b]:
                    c = C_LEFT if (a in LEFT and b in LEFT) else C_RIGHT if (a not in LEFT and b not in LEFT and a and b) \
                        else C_MID
                    if (a, b) in ((5, 6), (11, 12), (5, 11), (6, 12)):
                        c = C_MID
                    cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), c, lw, cv2.LINE_AA)
            r = max(2, int(3 * s))
            for i in np.where(vis)[0]:
                p = tuple(k[i, :2].astype(int))
                cv2.circle(img, p, r + 1, BLACK, -1, cv2.LINE_AA)
                cv2.circle(img, p, r, WHITE, -1, cv2.LINE_AA)
            # nhãn ID
            label = f"ID {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, fth)
            px, py = int(6 * s), int(5 * s)
            lx = min(max(x1, 0), W - tw - 2 * px)
            ly = max(y1, 0)
            top = ly - th - 2 * py if ly - th - 2 * py >= 0 else ly       # sát mép trên: nhãn vào trong khung
            cv2.rectangle(img, (lx, top), (lx + tw + 2 * px, top + th + 2 * py), color, -1)
            cv2.putText(img, label, (lx + px, top + th + py), cv2.FONT_HERSHEY_SIMPLEX, fscale, WHITE, fth, cv2.LINE_AA)
            if debug and debug.get(tid):
                self.draw_debug(img, tr["box"], debug[tid], max(top + th + 2 * py, max(y1, 0)) + 2)

    def _banner(self, img, e, n_active, now):
        """Băng cảnh báo mờ trên đầu video + viền đỏ nhấp nháy."""
        H, W = img.shape[:2]
        s = max(1.0, H / 540)
        bh = int(40 * s)
        ids = ", ".join(str(i) for i in e.get("ids", []))
        text = e["type"] + (f"  ·  ID {ids}" if ids else "") + f"  ·  {e.get('time', '')}"
        if n_active > 1:
            text += f"   (+{n_active - 1} cảnh báo khác)"
        key = (text, W, bh, e.get("severity"))
        if key != self.banner_key:
            _, bold = self._fonts(20 * s)
            col = sev_color(e.get("severity"))
            pil = Image.new("RGB", (W, bh), tuple(int(c * 0.75) for c in col))
            d = ImageDraw.Draw(pil)
            cx, cy, rr = int(bh * 0.55), bh // 2, int(bh * 0.3)                # dấu "!" trong vòng tròn
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(255, 255, 255))
            d.text((cx, cy), "!", font=bold, fill=col, anchor="mm")
            d.text((cx + rr + int(10 * s), cy), fit(bold, "CẢNH BÁO: " + text, W - cx - rr - int(20 * s)),
                   font=bold, fill=(255, 255, 255), anchor="lm")
            self.banner_key, self.banner_img = key, cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        roi = img[:bh]
        cv2.addWeighted(self.banner_img, 0.88, roi, 0.12, 0, dst=roi)
        pulse = 0.55 + 0.45 * abs(math.sin(now * math.pi * 1.5))
        col = sev_color(e.get("severity"))
        bgr = tuple(int(c * pulse) for c in col[::-1])
        t = max(4, int(6 * s))
        cv2.rectangle(img, (t // 2, t // 2), (W - 1 - t // 2, H - 1 - t // 2), bgr, t)

    # ================= panel =================
    def _panel(self, H, st, alerts, now, clock):
        s = max(1.0, H / 540)
        fs = self.font_size * s
        font, bold = self._fonts(fs)
        small, small_b = self._fonts(fs * 0.8)
        big, big_b = self._fonts(fs * 1.35)
        _, title_b = self._fonts(fs * 1.1)
        pw, pad = int(self.panel_width * s), int(14 * s)
        inner = pw - 2 * pad
        rad = int(8 * s)
        pil = Image.new("RGB", (pw, H), P_BG)
        d = ImageDraw.Draw(pil)

        # ---- tiêu đề ----
        hh = int(78 * s)
        d.rectangle([0, 0, pw, hh], fill=P_HEAD_BG)
        d.text((pad, pad), "GIÁM SÁT HÀNH VI", font=title_b, fill=P_TEXT)
        d.text((pw - pad, pad), clock, font=title_b, fill=P_TEXT, anchor="ra")
        active = any(now - e.get("t", 0) <= self.alert_sec for e in alerts)
        pill_t = "CÓ CẢNH BÁO" if active else "ĐANG GIÁM SÁT"
        pill_c = P_RED if active else P_GREEN
        py = pad + int(fs * 1.55)
        tw = small_b.getlength(pill_t)
        d.rounded_rectangle([pad, py, pad + tw + int(26 * s), py + int(fs * 1.25)], radius=int(fs * 0.62),
                            fill=tuple(int(c * 0.28) for c in pill_c), outline=pill_c, width=max(1, int(s)))
        cy = py + int(fs * 0.625)
        d.ellipse([pad + int(8 * s), cy - int(4 * s), pad + int(16 * s), cy + int(4 * s)], fill=pill_c)
        d.text((pad + int(20 * s), cy), pill_t, font=small_b, fill=pill_c, anchor="lm")
        src = st.get("source")
        if src is not None and str(src).isdigit():
            src = f"webcam {src}"
        if src:
            d.text((pw - pad, cy), fit(small, f"Nguồn: {src}", inner - tw - int(40 * s)), font=small, fill=P_DIM,
                   anchor="rm")
        y = hh + pad

        # ---- 4 ô số liệu ----
        cells = [("Camera", _fmt(st.get("cam_fps"), 1, " FPS")), ("Xử lý", _fmt(st.get("proc_fps"), 1, " FPS")),
                 ("Số người", "–" if st.get("people") is None else str(st["people"])),
                 ("YOLO", _fmt(st.get("yolo_ms"), 0, " ms"))]
        gap = int(8 * s)
        cw = (inner - gap) // 2
        ch = int(12 * s + fs * 0.95 + fs * 1.35 * 1.15)        # nhãn nhỏ + số to, không chồng nhau
        for i, (lab, val) in enumerate(cells):
            cx0 = pad + (i % 2) * (cw + gap)
            cy0 = y + (i // 2) * (ch + gap)
            d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + ch], radius=rad, fill=P_CARD)
            d.text((cx0 + int(10 * s), cy0 + int(6 * s)), lab, font=small, fill=P_DIM)
            d.text((cx0 + int(10 * s), cy0 + int(6 * s + fs * 0.95)), val, font=big_b, fill=P_TEXT)
        y += 2 * ch + gap + pad

        # ---- luồng 2 ----
        sc = st.get("scene")
        if sc is not None:
            score, thr = sc
            d.text((pad, y), "Luồng 2 · cảnh bất thường", font=small, fill=P_DIM)
            val = "chưa đủ 8 s" if score is None else f"{score:+.2f} / ngưỡng {thr:+.2f}"
            over = score is not None and score > thr
            d.text((pw - pad, y), val, font=small_b, fill=P_RED if over else P_TEXT, anchor="ra")
            y += int(fs * 1.25)
            bh = int(8 * s)
            lo, hi = thr - 10.0, thr + 4.0                       # thang hiển thị quanh ngưỡng
            d.rounded_rectangle([pad, y, pad + inner, y + bh], radius=bh // 2, fill=P_CARD)
            if score is not None:
                frac = float(np.clip((score - lo) / (hi - lo), 0.02, 1.0))
                d.rounded_rectangle([pad, y, pad + int(inner * frac), y + bh], radius=bh // 2,
                                    fill=P_RED if over else P_ACCENT)
            tx = pad + int(inner * (thr - lo) / (hi - lo))
            d.line([(tx, y - int(3 * s)), (tx, y + bh + int(3 * s))], fill=P_TEXT, width=max(1, int(2 * s)))
            y += bh + pad

        # ---- debug ----
        for line in st.get("debug") or []:                    # chừa chỗ cho tiêu đề cảnh báo + 1 thẻ + phím tắt
            if y + fs > H - pad - fs * 8.5:
                break
            d.text((pad, y), fit(small, line, inner), font=small, fill=(250, 204, 21))
            y += int(fs * 1.05)

        # ---- cảnh báo ----
        d.line([(pad, y), (pw - pad, y)], fill=P_SEP, width=1)
        y += pad
        d.text((pad, y), "CẢNH BÁO GẦN ĐÂY", font=bold, fill=P_TEXT)
        cnt = str(len(alerts))
        bw = small_b.getlength(cnt) + int(14 * s)
        d.rounded_rectangle([pw - pad - bw, y, pw - pad, y + int(fs * 1.2)], radius=int(fs * 0.6),
                            fill=P_RED if active else P_CARD)
        d.text((pw - pad - bw / 2, y + int(fs * 0.6)), cnt, font=small_b, fill=P_TEXT, anchor="mm")
        y += int(fs * 1.7)
        foot = H - pad - int(fs * 0.9)
        if not alerts:
            d.text((pad, y), "Chưa có cảnh báo", font=font, fill=P_FAINT)
        lh = int(fs * 1.3)
        for j, e in enumerate(alerts):                        # mới nhất trước
            new = now - e.get("t", 0) <= self.new_sec
            detail = e.get("detail") if new else None         # thẻ cũ thu gọn 2 dòng để hiện được nhiều hơn
            card_h = int(fs * 0.5) + lh + int(fs * 1.1) * (2 if detail else 1)
            more = int(fs * 1.1) if j < len(alerts) - 1 else 0    # chừa 1 dòng cho "… và N" nếu còn thẻ sau
            if y + card_h > foot - pad - more:
                d.text((pad, y), f"… và {len(alerts) - j} cảnh báo khác", font=small, fill=P_FAINT)
                break
            col = sev_color(e.get("severity"))
            d.rounded_rectangle([pad, y, pw - pad, y + card_h], radius=rad, fill=P_CARD_NEW if new else P_CARD)
            d.rounded_rectangle([pad, y, pad + int(5 * s), y + card_h], radius=int(3 * s), fill=col)
            x = pad + int(14 * s)
            ty = y + int(fs * 0.3)
            right = pw - pad - int(10 * s)
            if new:
                tag = "MỚI"
                tgw = small_b.getlength(tag) + int(12 * s)
                d.rounded_rectangle([right - tgw, ty + int(2 * s), right, ty + int(fs * 1.1)], radius=int(4 * s),
                                    fill=P_RED)
                d.text((right - tgw / 2, ty + int(fs * 0.58)), tag, font=small_b, fill=P_TEXT, anchor="mm")
                right -= tgw + int(8 * s)
            d.text((x, ty), fit(bold, e["type"], right - x), font=bold, fill=col if new else P_TEXT)
            ty += lh
            ids = ", ".join(str(i) for i in e.get("ids", []))
            sub = e.get("time", "") + (f"  ·  ID {ids}" if ids else "  ·  cả cảnh")
            if e.get("confidence") is not None:
                sub += f"  ·  {e['confidence']:.2f}"
            d.text((x, ty), fit(small, sub, pw - pad - x - int(10 * s)), font=small, fill=P_TEXT if new else P_DIM)
            if detail:
                ty += int(fs * 1.1)
                d.text((x, ty), fit(small, detail, pw - pad - x - int(10 * s)), font=small, fill=P_DIM)
            y += card_h + int(7 * s)

        # ---- phím tắt ----
        d.line([(pad, foot - int(6 * s)), (pw - pad, foot - int(6 * s))], fill=P_SEP, width=1)
        d.text((pw // 2, foot), "q thoát   ·   s chụp ảnh   ·   d debug", font=small, fill=P_FAINT, anchor="ma")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    # ================= ghép =================
    def draw(self, frame, tracks, status=None, alerts=(), highlight=(), now=None, debug=None):
        """Vẽ lên frame (tại chỗ; video thấp hơn min_height thì vẽ trên bản phóng to) và trả về ảnh mới
        = [video | panel].
        status: dict số liệu {cam_fps, proc_fps, yolo_ms, people, source, scene: (điểm|None, ngưỡng) | None,
                debug: [dòng]} (list chuỗi kiểu cũ vẫn nhận, hiện ở mục debug);
        alerts: list sự kiện (mục 5.2, có 'time', 't'), mới nhất trước;
        debug: {tid: [dòng chữ]} hiện trên khung từng người (chế độ debug) hoặc None."""
        now = time.time() if now is None else now
        H0, W0 = frame.shape[:2]
        if H0 < self.min_height:                              # video nhỏ (vd UCF 320x240): phóng lên, quy đổi toạ độ
            f = self.min_height / H0
            frame = cv2.resize(frame, (round(W0 * f), self.min_height), interpolation=cv2.INTER_LINEAR)
            tracks = {tid: {"box": tr["box"] * f,
                            "kpts": np.column_stack([tr["kpts"][:, :2] * f, tr["kpts"][:, 2]])}
                      for tid, tr in tracks.items()}
        alerts = list(alerts)
        st = dict(status) if isinstance(status, dict) else {"debug": list(status or [])}
        active = [e for e in alerts if now - e.get("t", 0) <= self.alert_sec]
        alert_ids = {i for e in active for i in e.get("ids", [])}
        self.draw_tracks(frame, tracks, highlight, debug, alert_ids)
        if active:
            top = max(active, key=lambda e: (e.get("severity") or 0, e.get("t", 0)))
            self._banner(frame, top, len(active), now)
        clock = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        key = (frame.shape[0], clock, tuple(sorted((k, repr(v)) for k, v in st.items())),
               tuple((e.get("t"), e["type"], now - e.get("t", 0) <= self.new_sec,
                      now - e.get("t", 0) <= self.alert_sec) for e in alerts))
        if key != self.cache_key:
            self.cache_key, self.cache_img = key, self._panel(frame.shape[0], st, alerts, now, clock)
        return np.hstack([frame, self.cache_img])
