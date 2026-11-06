"""Vẽ khung xương, ID lên video và panel bên phải (trạng thái + cảnh báo gần đây).

cv2.putText không hỗ trợ dấu tiếng Việt -> chữ có dấu vẽ bằng Pillow, chỉ trên
vùng panel (không chuyển đổi cả frame) để giữ tốc độ. Panel ghép CẠNH video nên
không che hình.
"""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
FONTS = [("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
         ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
         ("C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/tahomabd.ttf"),
         ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]

# màu cv2 (BGR) cho phần vẽ trên video
GREEN, RED, WHITE, YELLOW = (0, 220, 0), (0, 0, 255), (255, 255, 255), (0, 220, 255)
# màu Pillow (RGB) cho panel
P_BG, P_TEXT, P_DIM, P_HEAD = (28, 28, 32), (235, 235, 235), (150, 150, 150), (255, 210, 0)
P_NEW, P_SEP = (255, 70, 70), (70, 70, 75)


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


class UI:
    def __init__(self, font_size=18, kpt_conf=0.3, panel_width=360, new_sec=10.0):
        self.font_size, self.panel_width = font_size, panel_width
        self.kpt_conf, self.new_sec = kpt_conf, new_sec
        self.fonts = {}
        self.cache_key, self.cache_img = None, None         # panel chỉ vẽ lại khi nội dung đổi

    def _fonts(self, size):
        if size not in self.fonts:
            self.fonts[size] = (load_font(size), load_font(size, bold=True))
        return self.fonts[size]

    def draw_tracks(self, img, tracks, highlight=()):
        W = img.shape[1]
        for tid, tr in tracks.items():
            color = RED if tid in highlight else GREEN
            x1, y1, x2, y2 = tr["box"].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if tid in highlight else 2)
            k = tr["kpts"]
            vis = k[:, 2] >= self.kpt_conf
            for a, b in SKELETON:
                if vis[a] and vis[b]:
                    cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)),
                             YELLOW, 2, cv2.LINE_AA)
            for i in np.where(vis)[0]:
                cv2.circle(img, tuple(k[i, :2].astype(int)), 3, color, -1, cv2.LINE_AA)
            label = f"ID {tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx = min(max(x1, 0), W - tw - 6)
            ly = max(y1, 0)
            if ly - th - 8 >= 0:                              # đủ chỗ: nhãn nằm trên khung
                top = ly - th - 8
            else:                                             # sát mép trên: nhãn vào trong khung
                top = ly
            cv2.rectangle(img, (lx, top), (lx + tw + 6, top + th + 8), color, -1)
            cv2.putText(img, label, (lx + 3, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    def _panel(self, H, status, alerts, now):
        s = max(1.0, H / 540)                                 # phóng theo độ phân giải video
        fs = int(self.font_size * s)
        font, bold = self._fonts(fs)
        small, _ = self._fonts(int(fs * 0.85))
        pw, pad, lh = int(self.panel_width * s), int(12 * s), int(fs * 1.35)
        inner = pw - 2 * pad
        pil = Image.new("RGB", (pw, H), P_BG)
        d = ImageDraw.Draw(pil)

        y = pad
        for line in status:
            d.text((pad, y), fit(font, line, inner), font=font, fill=P_TEXT)
            y += lh
        y += pad // 2
        d.line([(pad, y), (pw - pad, y)], fill=P_SEP, width=1)
        y += pad

        d.text((pad, y), f"CẢNH BÁO GẦN ĐÂY ({len(alerts)})", font=bold, fill=P_HEAD)
        y += lh + pad // 2
        if not alerts:
            d.text((pad, y), "Chưa có cảnh báo", font=font, fill=P_DIM)
        for e in alerts:                                      # mới nhất trước
            if y + 2 * lh > H - pad:
                break
            new = now - e.get("t", 0) <= self.new_sec
            ids = ", ".join(str(i) for i in e.get("ids", []))
            sub = f"{e.get('time', '')}  ·  ID {ids}"
            if e.get("confidence") is not None:
                sub += f"  ·  {e['confidence']:.2f}"
            if e.get("detail"):
                sub += f"  ·  {e['detail']}"
            bar = int(4 * s)
            if new:
                d.rectangle([pad, y, pad + bar - 1, y + 2 * lh - int(6 * s)], fill=P_NEW)
            x = pad + bar + int(8 * s)
            d.text((x, y), fit(bold, e["type"], pw - x - pad), font=bold, fill=P_NEW if new else P_TEXT)
            d.text((x, y + lh), fit(small, sub, pw - x - pad), font=small, fill=P_TEXT if new else P_DIM)
            y += 2 * lh + int(4 * s)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def draw(self, frame, tracks, status=(), alerts=(), highlight=(), now=None):
        """Vẽ tracks lên frame (tại chỗ) và trả về ảnh mới = [video | panel].
        status: list chuỗi; alerts: list sự kiện (mục 5.2, có 'time', 't'), mới nhất trước."""
        self.draw_tracks(frame, tracks, highlight)
        now = time.time() if now is None else now
        alerts = list(alerts)
        key = (frame.shape[0], tuple(status),
               tuple((e.get("t"), e["type"], now - e.get("t", 0) <= self.new_sec) for e in alerts))
        if key != self.cache_key:
            self.cache_key, self.cache_img = key, self._panel(frame.shape[0], status, alerts, now)
        return np.hstack([frame, self.cache_img])
