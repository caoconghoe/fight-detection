import argparse, platform, shutil, time
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

MODELS = Path(__file__).resolve().parents[1] / "models"
MODELS.mkdir(exist_ok=True)
PT = MODELS / "yolov8n-pose.pt"
OV = MODELS / "yolov8n-pose_openvino_model"

def get_model(backend, imgsz):
    if not PT.exists():                              # tải lần đầu rồi chuyển vào models/
        YOLO("yolov8n-pose.pt")
        shutil.move("yolov8n-pose.pt", PT)
    if backend == "pt":
        return YOLO(str(PT))
    if not OV.exists():                              # export OpenVINO lần đầu
        YOLO(str(PT)).export(format="openvino", imgsz=imgsz)
    return YOLO(str(OV), task="pose")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["pt", "openvino"], default="pt")
    ap.add_argument("--source", default="0")         # 0 = webcam, hoặc URL điện thoại
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source

    model = get_model(a.backend, a.imgsz)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Khong mo duoc camera:", a.source)
        return

    times, n = [], 0
    while n < a.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = model.track(frame, imgsz=a.imgsz, persist=True, tracker="bytetrack.yaml",
                        device="cpu", verbose=False)[0]
        dt = time.perf_counter() - t0
        if n >= 10:                                  # bỏ 10 frame đầu (khởi động)
            times.append(dt)
        n += 1
        out = r.plot()
        cv2.putText(out, f"{a.backend} | {1 / dt:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("benchmark (q de thoat)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if times:
        t = np.array(times)
        print(f"\nBackend: {a.backend} | imgsz: {a.imgsz} | so frame do: {len(t)}")
        print(f"CPU: {platform.processor()}")
        print(f"Trung binh: {t.mean() * 1000:.1f} ms/frame  ->  {1 / t.mean():.1f} FPS (pose + tracking)")
        print(f"Cham nhat (p95): {np.percentile(t, 95) * 1000:.1f} ms")

if __name__ == "__main__":
    main()