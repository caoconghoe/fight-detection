"""Sinh khung xương tổng hợp (COCO-17) để test luật mà không cần camera."""
import math

import numpy as np

FPS = 15


def person(cx, hip_y, H=300, a_deg=0.0, arms="down", rw=None, re=None, noise=1.0, rng=None):
    """Một người cao H pixel, tâm hông (cx, hip_y), thân nghiêng a_deg độ (0 = đứng thẳng).
    arms: "down" | "up" | "head".  rw / re: vị trí cổ tay / khuỷu PHẢI tự đặt, dạng (a, b) =
    vai phải + a·H theo hướng ngang (dương = sang trái người) + b·H theo hướng lên của thân.
    Trả về {"box", "kpts"} như tracks (mục 5.1)."""
    rng = rng or np.random.default_rng()
    a = math.radians(a_deg)
    u = np.array([math.sin(a), -math.cos(a)])          # hông -> vai ("lên")
    p = np.array([math.cos(a), math.sin(a)])           # ngang
    hip = np.array([cx, hip_y], float)
    sh = hip + 0.3 * H * u
    nose = sh + 0.15 * H * u
    k = np.zeros((17, 2))
    k[0] = nose
    k[1], k[2] = nose + 0.03 * H * p + 0.02 * H * u, nose - 0.03 * H * p + 0.02 * H * u
    k[3], k[4] = nose + 0.06 * H * p, nose - 0.06 * H * p
    k[5], k[6] = sh + 0.12 * H * p, sh - 0.12 * H * p
    k[11], k[12] = hip + 0.08 * H * p, hip - 0.08 * H * p
    k[13], k[14] = k[11] - 0.25 * H * u, k[12] - 0.25 * H * u
    k[15], k[16] = k[13] - 0.25 * H * u, k[14] - 0.25 * H * u
    k[7], k[8] = k[5] - 0.15 * H * u, k[6] - 0.15 * H * u
    k[9], k[10] = k[7] - 0.15 * H * u, k[8] - 0.15 * H * u
    if arms == "up":
        k[7], k[8] = k[5] + 0.15 * H * u, k[6] + 0.15 * H * u
        k[9], k[10] = k[5] + 0.32 * H * u, k[6] + 0.32 * H * u
    elif arms == "head":
        k[9], k[10] = nose + 0.05 * H * p, nose - 0.05 * H * p
        k[7], k[8] = k[5] + 0.05 * H * u + 0.08 * H * p, k[6] + 0.05 * H * u - 0.08 * H * p
    if re is not None:
        k[8] = k[6] + re[0] * H * p + re[1] * H * u
    if rw is not None:
        k[10] = k[6] + rw[0] * H * p + rw[1] * H * u
    k += rng.normal(0, noise, k.shape)
    kp = np.hstack([k, np.full((17, 1), 0.9)]).astype(np.float32)
    m = 0.05 * H
    box = np.array([k[:, 0].min() - m, k[:, 1].min() - m, k[:, 0].max() + m, k[:, 1].max() + m], np.float32)
    return {"box": box, "kpts": kp}


def run(detector, frames_fn, secs, fps=FPS, frame_shape=None):
    """Chạy detector (hoặc EventManager) trên chuỗi tracks frames_fn(t) trong secs giây; trả về [(t, type, ids)].
    frame_shape (H, W): ảnh giả có kích thước này (để luật kiểm tra mép ảnh); None = không có ảnh."""
    frame = None if frame_shape is None else np.zeros((*frame_shape, 3), np.uint8)
    out = []
    for i in range(int(secs * fps)):
        t = i / fps
        evs = detector.update(frame, frames_fn(t), t)
        if isinstance(evs, tuple):                         # EventManager trả (events, highlight)
            evs = evs[0]
        out += [(round(t, 2), e["type"], e["ids"]) for e in evs]
    return out
