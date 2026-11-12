# Hệ thống phát hiện hành vi bất thường qua camera (ưu tiên đánh nhau)

Giám sát qua webcam / camera điện thoại / file video, **chạy realtime bằng CPU** (không cần GPU), cảnh báo cho người
giám sát khi phát hiện hành vi bất thường. Mỗi cảnh báo được lưu **ảnh chụp + clip 5 s + bản ghi SQLite** để duyệt lại.

| # | Hành vi | Cách phát hiện |
|---|---|---|
| 1 | **ĐÁNH NHAU** | Mô hình **Pair-TCN + MIL** trên khung xương từng cặp người (train trên RWF-2000 + RLVS) |
| 2 | TÉ NGÃ RỒI NẰM BẤT ĐỘNG | Nâng cấp khi NẰM BẤT ĐỘNG xảy ra ≤ 30 s sau TÉ NGÃ |
| 3 | HOẢNG LOẠN (đám đông bỏ chạy) | Luật tracking: nhiều người cùng chạy, tốc độ cảnh tăng vọt so với nền |
| 4 | TÉ NGÃ | Luật khung xương: hông rơi nhanh rồi thân nằm ngang |
| 5 | GIƠ TAY ĐẦU HÀNG | Luật tư thế: 2 tay giơ cao quá đầu ≥ 2.5 s |
| 6 | VẪY TAY CẦU CỨU | Luật tư thế: tay trên vai vẫy qua lại ≥ 5 lần trong 3 s |
| 7 | NẰM BẤT ĐỘNG | Luật tư thế: thân nằm ngang, gần như không cử động ≥ 10 s |
| 8 | ÔM ĐẦU / GỤC ĐẦU | Luật tư thế: 2 tay ôm đầu hoặc đầu gục thấp hơn vai ≥ 10 s |
| 9 | CHẠY NHANH | Luật khung xương: tốc độ ngang + tốc độ lại gần/ra xa camera (theo kích thước người), động tác chân, hướng ổn định |
| 10 | CẢNH BẤT THƯỜNG: <nhóm> | **Luồng 2**: MobileNetV3 + MIL train trên UCF-Crime, nhìn cả khung hình (cướp, cháy nổ, đột nhập, tai nạn…) |

Hai luồng chạy song song:
- **Luồng 1 – khung xương:** YOLOv8n-pose (OpenVINO, ~10–20 ms/frame) + ByteTrack trên **mọi frame** → chuỗi khung
  xương lấy mẫu **15 FPS theo thời gian** → các detector (loại 1–9).
- **Luồng 2 – cả cảnh:** luồng nền lấy **2 FPS**, MobileNetV3-Large (OpenVINO) → cửa sổ 16 frame (~8 s) → điểm bất
  thường + gợi ý nhóm (loại 10). Không làm giảm FPS luồng 1.

Kết quả 2 luồng vào `EventManager` (ưu tiên, nâng cấp, chống báo trùng) → giao diện OpenCV, SQLite, ảnh, clip.

---

## 1. Cài đặt

Yêu cầu: Windows 10/11, **Python 3.12**, webcam (hoặc camera điện thoại). Đặt thư mục dự án ở **đường dẫn không dấu**
(OpenCV trên Windows lỗi đọc/ghi file khi đường dẫn có dấu tiếng Việt).

```powershell
git clone https://github.com/caoconghoe/fight-detection.git D:\project\httm
cd D:\project\httm
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # nếu báo lỗi "running scripts is disabled":
                                    #   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt     # torch trên PyPI cho Windows là bản CPU
```

### Tạo mô hình YOLO OpenVINO (lần đầu)
Mô hình YOLO **không có trong git**. Tạo bằng:

```powershell
python realtime/tools/benchmark.py --backend openvino
```

Lệnh tự tải `yolov8n-pose.pt`, export OpenVINO (imgsz 416) vào `realtime/models/yolov8n-pose_openvino_model/`,
rồi mở webcam đo FPS (bấm `q` để thoát). `run.bat` cũng tự làm bước này nếu chưa có mô hình.

Các mô hình tự train **có sẵn trong repo** (`realtime/models/`), không cần tải thêm:
- Đánh nhau: `pair_tcn.onnx` + `fight_config.json`. Nếu thiếu, hệ thống tự dùng luật giả (2 người đứng gần + cổ tay
  chuyển động nhanh) để vẫn chạy được.
- Luồng 2: `scene_backbone.onnx`, `scene_head.onnx`, `scene_config.json`. Nếu thiếu, luồng 2 tự tắt.

---

## 2. Chạy

**Cách nhanh:** bấm đúp `run.bat` (kích hoạt `.venv` rồi chạy `realtime/main.py` với webcam).

Hoặc từ PowerShell (đã kích hoạt `.venv`, đứng ở thư mục dự án):

```powershell
python realtime/main.py                              # nguồn mặc định trong config.yaml (0 = webcam)
python realtime/main.py 0                            # webcam laptop
python realtime/main.py http://192.168.1.5:8080/video   # camera điện thoại (mục 3)
python realtime/main.py D:\video\test.mp4           # file video
run.bat http://192.168.1.5:8080/video                # run.bat cũng nhận nguồn như trên
```

Tuỳ chọn: `--no-show` (không mở cửa sổ), `--frames N` (dừng sau N frame), `--config <file.yaml>`.

### Phím tắt
| Phím | Chức năng |
|---|---|
| `q` / `Esc` / nút X | Thoát (phiên làm việc vẫn được đóng và lưu đầy đủ) |
| `s` | Chụp màn hình vào `realtime/events/snapshots/` |
| `d` | Bật/tắt **chế độ debug**: hiện trên khung mỗi người bộ đếm hiện tại của từng luật (để chỉnh ngưỡng) |

### Giao diện
- **Video (bên trái):** khung xương 2 màu (trái xanh dương, phải cam), ID mỗi người. Màu khung: xanh lá = bình thường,
  cam = đang trong một cảnh báo, **đỏ = cặp đang đánh nhau**. Khi có cảnh báo: **băng cảnh báo** trên đầu video và viền
  nhấp nháy trong 5 s (màu theo mức ưu tiên). Video nhỏ (vd 320×240) tự phóng lên 540 px để panel đủ chỗ.
- **Panel (bên phải):** đồng hồ + trạng thái (ĐANG GIÁM SÁT / CÓ CẢNH BÁO), 4 ô số liệu (FPS camera, FPS xử lý, số người,
  YOLO ms), thanh điểm luồng 2 so với ngưỡng, danh sách **CẢNH BÁO GẦN ĐÂY** dạng thẻ (nhãn MỚI trong 10 s đầu).
- Mỗi cảnh báo lưu vào `realtime/events/`: `events.db` (SQLite), `snapshots/*.jpg`, `clips/*.mp4` (3 s trước + 2 s sau).

---

## 3. Dùng camera điện thoại

Điện thoại và máy tính phải **cùng mạng Wi-Fi**. Đặt điện thoại nằm ngang, cố định (giá đỡ), độ phân giải 640×480 hoặc
1280×720 là đủ.

**IP Webcam (Android)**
1. Cài app *IP Webcam* → kéo xuống cuối → **Start server**.
2. App hiện địa chỉ, ví dụ `http://192.168.1.5:8080` → chạy `python realtime/main.py http://192.168.1.5:8080/video`.

**DroidCam (iPhone / Android) – phải dùng DroidCam Client trên Windows**
1. Trên điện thoại: mở app DroidCam, ghi lại *WiFi IP* và *Port* (mặc định 4747).
2. Trên máy tính: cài **DroidCam Client** (miễn phí, droidcam.com), mở ra, nhập đúng IP và Port, bấm *Start*.
   Điện thoại và máy tính phải cùng Wi-Fi.
3. DroidCam Client tạo một **webcam ảo**. Chạy `python realtime/main.py 1` (thử `0`, `2` nếu số khác là webcam laptop).

Lưu ý: DroidCam **không** phát luồng `http://…/video` cho trình duyệt như IP Webcam; địa chỉ `IP:4747` chỉ dùng cho DroidCam Client.

Nếu mất tín hiệu quá 3 s chương trình tự dừng và lưu phiên làm việc.

---

## 4. Duyệt sự kiện (`review.py`)

```powershell
python realtime/tools/review.py            # duyệt các sự kiện chưa review
python realtime/tools/review.py --stats    # chỉ in bảng thống kê
```

Mỗi sự kiện phát lặp clip (hoặc ảnh chụp) kèm thông tin. Phím: `y` = đúng, `n` = báo nhầm, `s` = bỏ qua,
`q` = thoát. Cuối cùng in bảng **precision theo từng loại** (số đúng / số đã duyệt).

---

## 5. Đánh giá trên thư mục video (`evaluate.py`)

```powershell
python realtime/tools/evaluate.py data_selfrecorded/test
```

Chạy đúng pipeline realtime (không hiển thị, không ghi DB) trên mọi video trong thư mục. **Nhãn lấy từ tiền tố tên file**
(phần trước dấu `_` đầu tiên):

| Tiền tố | Nhãn | Tiền tố | Nhãn |
|---|---|---|---|
| `fight_` | ĐÁNH NHAU | `handsup_` | GIƠ TAY ĐẦU HÀNG |
| `fall_` | TÉ NGÃ | `wave_` | VẪY TAY CẦU CỨU |
| `fallstill_` | TÉ NGÃ RỒI NẰM BẤT ĐỘNG | `still_`, `lying_` | NẰM BẤT ĐỘNG |
| `panic_` | HOẢNG LOẠN | `distress_`, `headhold_` | ÔM ĐẦU / GỤC ĐẦU |
| `run_`, `running_` | CHẠY NHANH | | |
| `nonfight_`, `normal_`, `neg_` | mẫu âm (không được báo gì) | khác | chạy nhưng không tính điểm |

Kết quả ghi vào `kaggle/results/`: `eval_<thư mục>_<thời gian>_clips.csv` (mỗi video: cảnh báo, thời điểm, EMA đánh
nhau lớn nhất, FPS, số người) và `..._metrics.csv` (precision / recall / F1 / độ trễ trung bình theo loại, mức clip).

### Công cụ hiệu chỉnh khác
```powershell
python realtime/tools/speed_probe.py <video>             # CHẠY NHANH: tốc độ ngang/dọc/tổng từng ID mỗi giây + CSV
python realtime/tools/scene_score_video.py <video|thư mục> [--compare]   # luồng 2: điểm từng cửa sổ, bảng + CSV
python realtime/tools/calibrate_scene.py 0 --minutes 15  # luồng 2: ghi điểm cảnh BÌNH THƯỜNG, đề xuất ngưỡng
```

---

## 6. Cấu hình (`realtime/config.yaml`)

| Mục | Nội dung |
|---|---|
| `source`, `camera` | Nguồn mặc định; backend webcam (`msmf` cho 30 FPS ổn định), FPS yêu cầu |
| `pose` | Mô hình YOLO, `imgsz 416`, `conf 0.3`, tracker `trackers/bytetrack_httm.yaml`, loại khung trùng |
| `sample_fps` | 15 – chuỗi khung xương đưa vào detector (phải khớp lúc train) |
| `events` | Cooldown 10 s, ưu tiên khi đang đánh nhau, nâng cấp té ngã |
| `full_body` | Điều kiện "thấy đủ thân" để xét té ngã / nằm bất động |
| `detectors` | Bật/tắt và ngưỡng từng detector (`fight`, `fall`, `pose_rules`, `scene`, `panic`, `running`) |
| `detectors.scene.threshold` | Ngưỡng luồng 2 (điểm là logit, không phải 0..1) – nên hiệu chỉnh bằng `calibrate_scene.py` trên camera thật |
| `ui` | Cỡ chữ, độ rộng panel, thời gian băng cảnh báo (`alert_sec`), chiều cao tối thiểu khi vẽ (`min_height`) |
| `recorder` | Clip quanh sự kiện: 3 s trước + 2 s sau, 15 FPS |

Tiêu chí ghép cặp người cho mô hình đánh nhau nằm **duy nhất** trong `realtime/shared/features.py::PARAMS`;
realtime ưu tiên `pair_params` ghi trong `fight_config.json` (tiêu chí lúc mô hình được train).

---

## 7. Mô hình đánh nhau và train lại trên Kaggle

- **Dữ liệu:** RWF-2000 (2000 clip, giữ chia train/val chính thức) + RLVS (2000 clip, chia 80/20).
- **Instance** = (cặp người, cửa sổ 32 frame mẫu ≈ 2.1 s) → 180 đặc trưng/frame (toạ độ, độ tin cậy, vận tốc khớp của
  2 người, khoảng cách cổ tay → đầu/thân người kia, khoảng cách 2 người). **Bag** = 1 clip, nhãn mức clip (MIL top-k).
- **Mô hình:** TCN 4 khối giãn nở (1/2/4/8), ~238k tham số, xuất ONNX (~1 MB), chạy ONNX Runtime CPU ~4 ms / lượt.

Quy trình (chi tiết trong notebook):
1. `kaggle/01_extract_pose` (trên Kaggle): trích khung xương 4000 clip → `poses/*.npz` + `meta.csv`.
2. `python realtime/tools/build_kaggle_notebook.py` → sinh `kaggle/02_train.ipynb` (chép nguyên `shared/sampling.py`,
   `shared/features.py` vào notebook để train và realtime dùng **cùng một code** trích đặc trưng).
3. Kaggle: File → Import Notebook → Add Input (output của 01) → GPU T4, Internet on → chạy `QUICK = True` thử,
   rồi `QUICK = False` → Save & Run All.
4. Tải `pair_tcn.onnx` + `fight_config.json` về `realtime/models/`, các file kết quả về `kaggle/results/`.

### Luồng 2 – cảnh bất thường (UCF-Crime)
- **Dữ liệu:** UCF-Crime (split chính thức: train 810 video bất thường + 800 Normal, test 140 + 150), thêm RWF NonFight
  và RLVS NonViolence làm mẫu bình thường.
- **Mô hình:** MobileNetV3-Large pretrained ImageNet (đóng băng) → 960 chiều/frame ở 2 FPS → cửa sổ 16 frame → MLP,
  học bằng ranking loss MIL (Sultani 2018) + head 5 nhóm (BẠO LỰC, CƯỚP, CHÁY NỔ, ĐỘT NHẬP/TRỘM, TAI NẠN).
- Notebook: `kaggle/04_scene_features.ipynb` (trích đặc trưng) và `kaggle/05_scene_train.ipynb` (train + xuất ONNX),
  sinh bởi `realtime/tools/build_scene_notebooks.py` và `build_scene_train.py`.

---

## 8. Kết quả hiện tại

**Mô hình đánh nhau (mức clip, tập val, ngưỡng 0.5)** – `kaggle/results/results.csv`:

| Train trên | Test | Accuracy | F1 | AUC |
|---|---|---|---|---|
| RWF | RWF val | 0.838 | 0.832 | 0.857 |
| RWF | RLVS val (chéo dataset) | 0.833 | 0.845 | 0.882 |
| RLVS | RWF val (chéo dataset) | 0.785 | 0.760 | 0.828 |
| RLVS | RLVS val | 0.918 | 0.915 | 0.932 |
| **Gộp** | **Gộp val** | **0.851** | **0.850** | **0.883** |
| Gộp | RWF val / RLVS val | 0.825 / 0.878 | 0.818 / 0.880 | 0.847 / 0.920 |

Giới hạn trên của recall (tỉ lệ clip đánh nhau có ≥ 1 cặp người được ghép): 0.90 (RWF val 0.875, RLVS val 0.93) –
clip mà YOLO chỉ bắt được 1 người thì không thể báo.

**Realtime trên 6 clip RWF val** (`evaluate.py`, imgsz 416): đúng 5/6 – không báo nhầm clip nào (precision 1.00),
báo 2/3 clip đánh nhau (recall 0.67), độ trễ báo trung bình 2.7 s.

**Luồng 2 (UCF-Crime test):** AUC mức video 0.906, AUC mức cửa sổ 0.749, báo nhầm trên video Normal 7.3%; phân nhóm
đúng ~40% (tên nhóm chỉ là gợi ý).

**Tốc độ (Intel i5-13500H, CPU):** webcam 640×480 ~30 FPS; video 1280×720 có 4–9 người ~19 FPS
(giới hạn bởi YOLO + giải mã video; mô hình đánh nhau chỉ thêm vài ms mỗi lượt).

---

## 9. Kiểm thử

```powershell
python realtime/tests/test_features.py     # trích đặc trưng (không phụ thuộc FPS gốc, đối xứng, ghép cặp)
python realtime/tests/test_fight.py        # detector đánh nhau (luật giả + nhánh ONNX với mô hình giả)
python realtime/tests/test_pose_rules.py   # giơ tay, vẫy tay, vò đầu...
python realtime/tests/test_body_rules.py   # chống báo nhầm khi ngồi gần camera, té ngã toàn thân
python realtime/tests/test_recorder.py     # clip quanh sự kiện
python realtime/tests/test_evaluate.py     # nhãn + precision/recall của evaluate.py
python realtime/tests/test_running.py      # chạy nhanh (chạy ngang/lại gần/ra xa, đi bộ, đứng dậy, nhảy, đánh nhau...)
python realtime/tests/test_scene.py        # luồng 2: nhận lớp, chia tập, tiền xử lý ảnh
python realtime/tests/test_scene_detector.py   # luồng 2: logic cảnh báo
```

---

## 10. Cấu trúc thư mục

```
httm/
├── run.bat, README.md, requirements.txt, CLAUDE.md (đặc tả + tiến độ)
├── kaggle/            02_train / 04_scene_features / 05_scene_train .ipynb (sinh tự động), results/
├── data_selfrecorded/ video tự quay / video kiểm tra (không đưa lên git)
└── realtime/
    ├── main.py, config.yaml
    ├── core/          camera, pose (YOLO + ByteTrack), ui, events (EventManager), db (SQLite), recorder
    ├── detectors/     base, fight (Pair-TCN ONNX), fall, pose_rules, panic, running, scene (luồng 2)
    ├── shared/        sampling.py, features.py, scene.py – dùng chung realtime + Kaggle
    ├── trackers/      bytetrack_httm.yaml
    ├── models/        pair_tcn.onnx, fight_config.json, scene_*.onnx, scene_config.json (+ YOLO tự tạo, không có trong git)
    ├── tools/         benchmark, review, evaluate, speed_probe, scene_score_video, calibrate_scene, build_*_notebook
    ├── tests/         unit test + dữ liệu giả
    └── events/        events.db, snapshots/, clips/ (sinh ra khi chạy, không đưa lên git)
```

---

## 11. Hạn chế và hướng phát triển

- Cảnh đông người (camera giám sát) ByteTrack hay đổi ID → điểm đánh nhau theo từng cặp bị ngắt quãng, báo chậm hoặc sót.
- Ngồi sát webcam không thấy đủ thân → luật té ngã / nằm bất động tạm bỏ qua (có chủ đích, tránh báo nhầm).
- Đang làm: fine-tune bằng video tự quay (đánh nhau dàn dựng + mẫu âm khó: ôm nhau, đập tay, vươn vai...),
  đánh giá luật trên UR Fall / Le2i (té ngã) và UMN (hoảng loạn).
- Luồng 2: điểm trên video ngoài UCF-Crime (camera khác, video tải từ mạng) có thể rất thấp → ngưỡng phải hiệu chỉnh
  theo từng camera; cần video ≥ 7.5 s mới có cửa sổ đầu tiên.
- Hướng phát triển: đồ vật bị bỏ lại, cháy/khói bằng YOLO riêng, vũ khí, backend web xem lịch sử cảnh báo.
