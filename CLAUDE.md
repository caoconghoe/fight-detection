# CLAUDE.md – Hệ thống phát hiện hành vi bất thường (ưu tiên đánh nhau)

> File này là **bản giao việc đầy đủ** cho Claude Code. Đọc hết trước khi làm.
> Người dùng là sinh viên, giao tiếp bằng **tiếng Việt** → luôn trả lời bằng tiếng Việt.
> Làm **từng bước nhỏ**, xong bước nào thì chạy thử / kiểm tra rồi mới sang bước sau.
> Mọi quyết định kỹ thuật dưới đây **đã được chốt** – không tự ý đổi hướng (dataset, mô hình, FPS, kiến trúc) nếu chưa hỏi người dùng.

---

## 1. Tổng quan dự án

**Mục tiêu:** hệ thống giám sát qua camera, cảnh báo cho người giám sát khi có hành vi bất thường. **Ưu tiên số 1 là phát hiện đánh nhau.**

**Ràng buộc:**
- Máy người dùng **không có GPU** (Intel Core i5-13500H, Windows 11). Chạy realtime **bằng CPU**.
- **Train trên Kaggle** (GPU T4), tải file mô hình (`.onnx`) về chạy trên máy.
- Nguồn video: **webcam laptop** hoặc **camera điện thoại** (app IP Webcam: `http://<ip>:8080/video`) hoặc file video (để kiểm thử).
- Giao diện: **cửa sổ OpenCV chạy local** (KHÔNG dùng Gradio / web server – đã thử Gradio bị giật).
- Dùng **ít nhất 2 dataset** để train mô hình đánh nhau.
- Lưu sự kiện bằng **SQLite**.

**8 loại bất thường cần phát hiện:**

| # | Loại | Cách phát hiện | Ưu tiên |
|---|---|---|---|
| 1 | ĐÁNH NHAU | Mô hình **Pair-TCN + MIL** (train có giám sát yếu) | 100 |
| 2 | TÉ NGÃ RỒI NẰM BẤT ĐỘNG | Kết hợp luật té ngã + nằm bất động | 90 |
| 3 | HOẢNG LOẠN (đám đông bỏ chạy) | Luật tracking | 85 |
| 4 | TÉ NGÃ | Luật khung xương | 80 |
| 5 | GIƠ TAY ĐẦU HÀNG | Luật tư thế | 70 |
| 6 | VẪY TAY CẦU CỨU | Luật tư thế | 60 |
| 7 | NẰM BẤT ĐỘNG | Luật tư thế | 60 |
| 8 | ÔM ĐẦU / GỤC ĐẦU | Luật tư thế | 40 |
| 9 | CẢNH BẤT THƯỜNG: <nhóm> | **Luồng 2** – ảnh màu cả cảnh, MobileNetV3 + MIL train trên UCF-Crime (mục 12) | 50 |

Hệ thống có **2 luồng chạy song song**: **luồng 1** = khung xương (YOLO-pose → loại 1–8, mục 5–9);
**luồng 2** = nhìn cả cảnh (mục 12) – bắt bất thường không thể hiện rõ qua tư thế (cướp, cháy nổ, đột nhập, tai nạn...).

**Để ở mục "hướng phát triển" (KHÔNG làm bây giờ):** đồ vật bị bỏ lại, vũ khí, backend Spring Boot xem lịch sử cảnh báo.
**Việc sau (đã ghi nhận, CHƯA làm):** phát hiện cháy/khói bằng **YOLO riêng** (luồng 2 chỉ cho nhóm CHÁY NỔ ở mức cả cảnh).

---

## 2. Môi trường & thông số đã chốt

- Thư mục dự án: `D:\project\httm` (đường dẫn **không dấu** – bắt buộc, vì OpenCV trên Windows lỗi đọc/ghi file khi đường dẫn có dấu tiếng Việt).
- Python **3.12.10** trong `.venv` (máy có cả 3.14 nhưng KHÔNG dùng). Kích hoạt: `.venv\Scripts\Activate.ps1`.
- Thư viện chính: `ultralytics 8.4.x`, `torch 2.15 (CPU)`, `opencv-python 5.0`, `openvino 2026.4`, `onnxruntime 1.31`, `pillow`, `pyyaml`, `numpy 2.x`, `lap`.
- Cập nhật requirements bằng: `pip freeze | Out-File -Encoding utf8 requirements.txt` (KHÔNG dùng `>` vì PowerShell ghi UTF-16).
- Terminal là **PowerShell**.

**Kết quả đo FPS (YOLOv8n-pose + ByteTrack, imgsz 416, CPU):**

| Backend | ms/frame | FPS |
|---|---|---|
| PyTorch `.pt` | 44.4 | 22.5 |
| **OpenVINO** | **21.5** | **46.5** |

**Thông số chốt:**
- Mô hình pose realtime: `realtime/models/yolov8n-pose_openvino_model/`, `imgsz=416`, `conf=0.3`, tracker `realtime/trackers/bytetrack_httm.yaml` (track_buffer 60).
- **Mô hình YOLO OpenVINO KHÔNG có trong git** – tự tạo bằng: `python realtime/tools/benchmark.py --backend openvino`
  (tự tải `yolov8n-pose.pt` nếu chưa có, export OpenVINO imgsz 416 vào `realtime/models/`, rồi đo FPS trên webcam; bấm `q` để thoát).
- Mô hình đánh nhau `realtime/models/pair_tcn.onnx` + `fight_config.json` **có trong git** (sinh từ notebook `kaggle/02_train.ipynb`).
- YOLO + tracking chạy **mọi frame camera** (~30 FPS) để ID ổn định.
- **`sample_fps = 15`**: chuỗi khung xương đưa vào mô hình/luật được **lấy mẫu theo thời gian về đúng 15 FPS** (dùng `shared/sampling.py::FpsSampler`). Train trên Kaggle cũng phải lấy mẫu đúng như vậy. Lý do: RLVS có clip chỉ ~22 FPS, mô hình học vận tốc/gia tốc nên FPS train và chạy thật phải khớp.

---

## 3. Cấu trúc thư mục (mục tiêu)

```
httm/
├── CLAUDE.md                 # file này
├── requirements.txt
├── .gitignore
├── run.bat                   # (tuần 7) bấm đúp để chạy
├── README.md                 # (tuần 7)
├── kaggle/
│   ├── 01_extract_pose.ipynb # ĐÃ CÓ trên Kaggle (đang chạy) – tải bản về lưu ở đây
│   ├── 02_train.ipynb        # ĐÃ CÓ (luồng 1) – sinh bởi tools/build_kaggle_notebook.py
│   ├── 04_scene_features.ipynb # luồng 2: trích đặc trưng MobileNetV3 – sinh bởi tools/build_scene_notebooks.py
│   ├── 05_scene_train.ipynb  # luồng 2: train MIL + xuất ONNX (cần viết)
│   └── results/              # bảng kết quả train / đánh giá
├── data_selfrecorded/        # video nhóm tự quay (gitignore)
└── realtime/
    ├── main.py               # ĐÃ CÓ
    ├── config.yaml           # ĐÃ CÓ – cần mở rộng thêm ngưỡng các detector
    ├── core/
    │   ├── camera.py         # ĐÃ CÓ
    │   ├── pose.py           # ĐÃ CÓ
    │   ├── ui.py             # ĐÃ CÓ
    │   ├── events.py         # cần viết: EventManager
    │   ├── db.py             # cần viết: SQLite
    │   └── recorder.py       # cần viết: lưu clip quanh sự kiện
    ├── detectors/
    │   ├── base.py           # cần viết: lớp Detector
    │   ├── fight.py          # cần viết: Pair-TCN (tạm dùng luật giả khi chưa có .onnx)
    │   ├── fall.py           # cần viết
    │   ├── pose_rules.py     # cần viết
    │   ├── panic.py
    │   └── scene.py          # luồng 2 (cần viết): luồng nền 2 FPS, OpenVINO
    ├── shared/
    │   ├── sampling.py       # ĐÃ CÓ – DÙNG CHUNG với Kaggle
    │   ├── features.py       # DÙNG CHUNG với Kaggle (luồng 1)
    │   └── scene.py          # DÙNG CHUNG với Kaggle (luồng 2): hằng số, nhận lớp, tiền xử lý ảnh
    ├── models/               # yolov8n-pose.pt, *_openvino_model/, pair_tcn.onnx, fight_config.json,
    │                         # scene_backbone.onnx, scene_head.onnx, scene_config.json (luồng 2)
    ├── tools/
    │   ├── benchmark.py      # ĐÃ CÓ
    │   ├── review.py         # cần viết
    │   ├── evaluate.py       # cần viết
    │   ├── build_kaggle_notebook.py  # sinh 02_train.ipynb (mục 7)
    │   └── build_scene_notebooks.py  # sinh 04/05 (mục 12)
    └── events/               # ảnh, clip, events.db sinh ra khi chạy (gitignore)
```

Lệnh chạy (từ thư mục `httm`): `python realtime/main.py [nguồn]` – nguồn có thể là `0`, URL điện thoại hoặc đường dẫn video.

---

## 4. Trạng thái hiện tại (cập nhật 2026-11-10)

**Đã xong (code trong repo là nguồn chính xác nhất; mục 11 chỉ là bộ khung tuần 1 ban đầu, nay đã mở rộng):**
- Môi trường `.venv` Python 3.12, đo FPS, export OpenVINO. Repo GitHub `caoconghoe/fight-detection`.
  `.gitignore` bỏ `.venv/`, `__pycache__/`, `realtime/models/*.pt`, `realtime/models/*_openvino_model/`,
  `realtime/models/*.onnx` trừ `pair_tcn.onnx`, `realtime/events/`, `data_selfrecorded/`.
- **Giai đoạn A (A0–A6):** bộ khung (webcam MSMF 30 FPS, panel cảnh báo, phím `s`/`d`), `EventManager`, SQLite,
  4 detector (đánh nhau, té ngã, luật tư thế, hoảng loạn), recorder clip, `review.py`. Đã sửa theo thử thật:
  vẫy tay (lọc tay gần đầu, ngưỡng đổi hướng 0.15 thân), giơ tay (khuỷu/cổ tay/đỉnh đầu ước lượng/cách đầu),
  té ngã + nằm im chỉ xét khi thấy đủ thân (mục `full_body` trong config), thang đo thân chống co ngắn khi cúi,
  tracker riêng `realtime/trackers/bytetrack_httm.yaml` (track_buffer 60), loại khung trùng.
- **Giai đoạn B (B1–B3):** `shared/features.py` (tiêu chí ghép cặp trong `PARAMS`, mặc định = mức 7),
  `tools/build_kaggle_notebook.py` → `kaggle/02_train.ipynb`, đã train đầy đủ trên Kaggle (4000 clip).
  **Mốc B đạt:** gộp → gộp val **F1 0.850, AUC 0.883** (RWF val F1 0.818, RLVS val 0.880); mức tiêu chí 7
  (có mặt chung 30%, nối tracklet 1 s); 90% clip Fight val có ≥ 1 instance. Kết quả: `kaggle/results/`.
- **Giai đoạn C (C1):** `detectors/fight.py` dùng `pair_tcn.onnx` + `fight_config.json` (pair_params, mean/std,
  threshold 0.6), ONNX Runtime 2 luồng (~4 ms/lượt). Kiểm tra 6 clip RWF val (`data_selfrecorded/test/`):
  **đúng 5/6** (3/3 NonFight không báo, 2/3 Fight báo ở 2.1 s và 3.4 s; clip sót do ByteTrack đổi ID liên tục và
  mô hình chỉ cho điểm cao ở giây cuối). Webcam 640×480 vẫn ~30 FPS khi bật mô hình.
- **D2:** `tools/evaluate.py` (nhãn theo tiền tố tên file, CSV vào `kaggle/results/`). **D3:** `run.bat`, `README.md`.
- Unit test: `realtime/tests/test_*.py` (features, fight, pose_rules, body_rules, recorder, evaluate).

**Đang làm – LUỒNG 2 (mục 12, quyết định 2026-11-08):** đã có `shared/scene.py`, mục `scene` trong config,
`kaggle/04_scene_features.ipynb` (chạy thử trên máy bằng dữ liệu giả: quét, gộp 2 bản UCF bỏ trùng theo tên file,
chia split, trích, chạy tiếp, ước tính thời gian đều đúng). **2026-11-10:** chuyển sang bản UCF đầy đủ
`anomalydetectiondatasetucf` + split chính thức (mục 12.2); chờ người dùng chạy QUICK trên Kaggle. Notebook 05 CHƯA có
trong repo (yêu cầu ở mục 12.4). Notebook 03 trích khung xương UCF: **tạm dừng / không làm**.

**Còn lại:**
- Thử webcam với **2 người thật** (Mốc C): giả vờ đánh nhau → báo đúng cặp, khung đỏ; mẫu âm khó (ôm, đập tay, bắt tay).
- Thử **té ngã toàn thân** (camera thấy cả người) và ÔM ĐẦU ở khoảng cách xa (hiện chỉ báo khi ngồi gần camera).
- **D1:** fine-tune bằng video tự quay + script trích khung xương trên Kaggle cho `data_selfrecorded/`.
- Đánh giá luật trên UR Fall / Le2i (té ngã), UMN (hoảng loạn) bằng `evaluate.py` (đổi tên file theo tiền tố).

---

## 5. Giao diện dữ liệu chung (BẮT BUỘC tuân theo)

### 5.1 `tracks` (đầu ra `core/pose.py::PoseTracker`)
```python
tracks = {track_id: {"box": np.ndarray(4,) xyxy pixel, "kpts": np.ndarray(17, 3) [x_pixel, y_pixel, conf]}}
```
Keypoint theo chuẩn COCO-17: 0 mũi, 1-2 mắt (L,R), 3-4 tai (L,R), 5-6 vai, 7-8 khuỷu, 9-10 cổ tay, 11-12 hông, 13-14 gối, 15-16 cổ chân. Chỉ số chẵn/lẻ: lẻ = trái, chẵn = phải (trừ 0).

### 5.2 Sự kiện (event)
```python
{"type": "ĐÁNH NHAU", "severity": 100, "ids": [3, 7], "confidence": 0.87 | None}
```
`main.py` sẽ thêm `time` (HH:MM:SS) và `t` (time.time()) trước khi hiển thị/lưu.

### 5.3 Lớp `Detector` (`detectors/base.py`)
```python
class Detector:
    name = "base"
    def update(self, frame, tracks, t):
        """Gọi ở MỖI frame được lấy mẫu (15 FPS – khi sampler.tick(t) == True).
        frame: ảnh BGR; tracks: như 5.1; t: thời điểm (giây).
        Trả về list sự kiện (5.2) vừa XÁC NHẬN (không trả lặp mỗi frame)."""
        return []
    def highlight_ids(self):
        """(tuỳ chọn) ID cần tô đỏ trên màn hình, vd cặp đang đánh nhau."""
        return set()
```
Mọi detector giả định dữ liệu vào ở **15 FPS** (đọc từ `config.yaml: sample_fps`). Detector tự xoá dữ liệu của ID không còn xuất hiện.

### 5.4 `EventManager` (`core/events.py`)
- Giữ danh sách detector, gọi `update` của từng cái trong mỗi tick lấy mẫu.
- **Ưu tiên**: sắp xếp sự kiện theo bảng ưu tiên ở mục 1. Nếu một ID đã nằm trong sự kiện ưu tiên cao hơn **trong cùng tick hoặc đang có cảnh báo ĐÁNH NHAU còn hiệu lực (≤ 3 s)**, bỏ các sự kiện ưu tiên thấp hơn của ID đó (vd người đang đánh nhau giơ tay đỡ đòn → không báo GIƠ TAY).
- **Nâng cấp**: nếu có NẰM BẤT ĐỘNG cho ID đã có TÉ NGÃ trong 30 s gần nhất → đổi type thành `TÉ NGÃ RỒI NẰM BẤT ĐỘNG`.
- **Cooldown**: cùng (type, tập ids) không báo lại trong 10 s.
- Trả về danh sách sự kiện cuối cùng + tập ID cần tô đỏ.

```python
PRIORITY = {"ĐÁNH NHAU": 100, "TÉ NGÃ RỒI NẰM BẤT ĐỘNG": 90, "HOẢNG LOẠN": 85,
            "TÉ NGÃ": 80, "GIƠ TAY ĐẦU HÀNG": 70, "VẪY TAY CẦU CỨU": 60,
            "NẰM BẤT ĐỘNG": 60, "CẢNH BẤT THƯỜNG": 50, "ÔM ĐẦU / GỤC ĐẦU": 40}
```
Sự kiện luồng 2 có `type = "CẢNH BẤT THƯỜNG: <nhóm>"` (vd `CẢNH BẤT THƯỜNG: CƯỚP`), `ids = []` (cả cảnh);
mức ưu tiên tra theo phần trước dấu ":" (= 50). Cooldown tính riêng từng nhóm (vì type khác nhau).

### 5.5 SQLite (`core/db.py`) – dùng đúng schema này
```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL, ended_at TEXT, source TEXT, config_json TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts TEXT NOT NULL, type TEXT NOT NULL, severity INTEGER,
    track_ids TEXT,              -- JSON, vd "[3, 7]"
    confidence REAL, snapshot_path TEXT, clip_path TEXT,
    reviewed INTEGER DEFAULT 0,
    is_true INTEGER,             -- 1 đúng, 0 báo nhầm, NULL chưa duyệt
    note TEXT);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""
```
**Luồng 2 bổ sung 3 cột** vào `events` (DB cũ tự `ALTER TABLE ... ADD COLUMN` khi mở): `scene_score REAL`
(điểm bất thường luồng 2 lúc báo), `scene_group TEXT` (nhóm luồng 2 lúc báo), `fight_score REAL` (EMA đánh nhau
lớn nhất của luồng 1 lúc báo). **Mọi** sự kiện (của cả 2 luồng) đều ghi đủ 3 cột để so sánh.

Lớp `EventDB(path="realtime/events/events.db")`: `PRAGMA journal_mode=WAL`, `check_same_thread=False` + `threading.Lock`. Hàm: `start_session(source, config)`, `end_session(id)`, `add_event(session_id, type, severity, ids, confidence, snapshot, clip) -> id`, `set_clip(event_id, path)`, `review(event_id, is_true, note)`, `unreviewed()`, `stats()` (theo type: total, reviewed, correct, precision).

---

## 6. Dữ liệu trên Kaggle (đầu ra notebook `01_extract_pose`)

**Dataset dùng để train đánh nhau (2 bộ):**
- RWF-2000: `/kaggle/input/datasets/daniazehra/rwf-2000/RWF-2000/{train,val}/{Fight,NonFight}/*.avi` – 800+800 train, 200+200 val, 30 FPS, ~150 frame/clip. **Giữ chia train/val chính thức.**
- RLVS: `/kaggle/input/datasets/mohamedmustafa/real-life-violence-situations-dataset/Real Life Violence Dataset/{Violence,NonViolence}/*.mp4` – 1000+1000, FPS 22–30. Thư mục `real life violence situations/...` là **bản sao trùng → KHÔNG dùng**. Chia 80/20 stratified, `random_state=42`.
- Tổng 4000 clip.

**Cách trích xuất:** YOLOv8n-pose, `imgsz=640` (offline, để bắt người nhỏ ở camera giám sát), `conf=0.25`, ByteTrack, `persist=False` mỗi video, **mọi frame ở FPS gốc**, tối đa 20 s/clip.

**Định dạng file** `/kaggle/working/poses/{dataset}__{split}__{tên}__{label}.npz`:
- `data`: float32 (N, 57), mỗi dòng 1 người ở 1 frame: `[frame_idx, track_id, x1, y1, x2, y2, 17 × (x, y, conf)]`. **Toạ độ chuẩn hoá 0..1 theo W và H riêng biệt** (box là `xyxyn`). `track_id = -1` nếu tracker chưa gán.
- `fps`, `n_frames`, `width`, `height`.
- ⚠️ Khi tính đặc trưng phải **nhân lại `x*width`, `y*height`** để về pixel (tránh méo tỉ lệ khung hình), rồi mới chuẩn hoá theo chiều cao người như realtime.

**`/kaggle/working/meta.csv`** cột: `file, dataset, label, split, src, n_frames, max_persons, n_long_tracks, fps, has_pair`.
Giá trị: `dataset` ∈ {`rwf2000`, `rlvs`}; `label` số nguyên 0/1 (1 = Fight/Violence); `split` ∈ {`train`, `val`};
`file` = `{dataset}__{split}__{tên}__{label}.npz` (tên file, nằm trong `poses/`). Notebook 01 đã chạy xong (Version #1, 4000 clip).

**Quan sát từ chạy thử 20 clip:** ~1.6 s/clip; khung xương khớp người; khi 2 người quấn nhau YOLO hay chỉ bắt 1 người; `has_pair` (≥2 track giữ ID ≥30% clip) khá thấp (RWF Fight ~0.6) do ID hay nhảy → cần **nối tracklet** và **xét cặp theo cửa sổ** (mục 7.1), KHÔNG lọc bỏ clip theo `has_pair`.

---

## 7. Đặc tả mô hình đánh nhau (Pair-TCN + MIL)

### 7.1 `shared/features.py` (dùng chung Kaggle + realtime – một nguồn duy nhất)
Viết hàm thuần numpy, không phụ thuộc ultralytics/cv2.

1. **Chuyển dữ liệu** về dạng theo track: `{tid: list[(t_giây, box_pixel(4), kpts_pixel(17,3))]}`.
2. **Lấy mẫu 15 FPS** bằng logic `FpsSampler` (theo thời gian `t = frame_idx / fps`).
3. **Nối tracklet** (chỉ cần ở offline; realtime giữ ID bằng ByteTrack persist): track A kết thúc tại t, track B bắt đầu trong ≤ 1.0 s, khoảng cách tâm box < 1.0 × chiều cao box → gộp B vào A. Detection track `-1` → gán vào track gần nhất nếu IoU > 0.5, không thì bỏ.
4. **Lấp frame thiếu** của một người trong cửa sổ: nội suy tuyến tính nếu khoảng trống ≤ 8 frame mẫu, ngược lại để 0 và conf 0. Clip < 32 frame mẫu: 1 cửa sổ đệm 0; thêm cửa sổ cuối căn về cuối clip.
5. **Cửa sổ:** `WINDOW = 32` frame mẫu (~2.1 s), bước trượt `STRIDE = 8`.
6. **Chọn cặp ứng viên trong cửa sổ:** 2 người cùng có mặt ≥ 30% frame; tại ít nhất 1 frame khoảng cách tâm box ≤ 2.0 × chiều cao trung bình. Tối đa 6 cặp gần nhau nhất / cửa sổ.
   **Mọi tiêu chí nằm DUY NHẤT trong `features.PARAMS`** (mức 7 của bảng quét notebook 02). Realtime ưu tiên `pair_params` trong `fight_config.json`.
7. **Đặc trưng mỗi frame của cặp (A, B)** – chuẩn hoá: gốc toạ độ = trung điểm tâm hông của A và B (nếu thiếu hông dùng tâm box), thang đo = chiều cao box trung bình của 2 người. Keypoint conf < 0.3 → toạ độ 0, conf 0.
   - (x, y) 17 khớp của A và B → 68
   - conf 17 khớp của A và B → 34
   - vận tốc (x, y) 17 khớp A và B (hiệu frame mẫu liên tiếp, đã chuẩn hoá) → 68
   - khoảng cách mỗi cổ tay của A tới đầu B và tới tâm thân B, và ngược lại → 8
   - khoảng cách tâm A–B và tốc độ thay đổi của nó → 2
   - **Tổng FEAT_DIM = 180.** Hàm trả về mảng `(WINDOW, 180)` float32.
8. Hàm public gợi ý:
   - `tracks_from_npz(npz) -> dict` (offline)
   - `build_pair_windows(tracks_by_id, window=32, stride=8) -> list[(idA, idB, start_t, feats(32,180))]`
   - `pair_features(seqA, seqB) -> np.ndarray(32,180)` (dùng cả realtime)

### 7.2 Mô hình (PyTorch, train trên Kaggle)
- **Instance** = (cặp, cửa sổ) → `(32, 180)`. **Bag** = 1 clip (nhãn Fight/NonFight).
- **TCN:** Conv1d 180→128, 4 block residual (kernel 3, dilation 1/2/4/8, BatchNorm, ReLU, dropout 0.2), pool max + mean theo thời gian → Linear 256→64→1 (logit).
- **MIL pooling:** điểm bag = trung bình top-k logit instance, `k = max(1, ceil(n/8))`. Loss BCEWithLogits trên bag.
- Clip nhãn 1 không có instance nào → bỏ khi train (in số lượng). Clip nhãn 0 không có instance → dự đoán 0 khi đánh giá.
- **Augment:** nhiễu Gauss σ=0.01 lên toạ độ; xoá ngẫu nhiên 10% khớp; 30% xác suất xoá toàn bộ thân dưới (khớp 11–16, mô phỏng webcam thấy nửa người); lật ngang (x→−x và hoán đổi khớp trái/phải); hoán đổi A/B; co giãn thời gian 0.8–1.2 bằng nội suy.
- Chuẩn hoá đặc trưng bằng mean/std tính trên instance tập train.
- AdamW lr 1e-3, weight decay 1e-4, cosine, ~40 epoch, batch 16 bag. Lưu checkpoint tốt nhất theo F1 val.
- **Đánh giá (cho báo cáo):** Accuracy, F1, AUC mức clip trên val của từng dataset và gộp; thí nghiệm **train RWF → test RLVS**, **train RLVS → test RWF**, **train gộp**.
- **Xuất:** `pair_tcn.onnx` (input `x`: `(batch, 32, 180)`, dynamic batch, output `logit` theo instance, opset 17) + `fight_config.json`:
  ```json
  {"fps": 15, "window": 32, "stride": 8, "feat_dim": 180, "mean": [...], "std": [...], "threshold": 0.6}
  ```

### 7.3 Đưa code chung lên Kaggle
Viết `realtime/tools/build_kaggle_notebook.py` sinh ra `kaggle/02_train.ipynb`: ô đầu dùng `%%writefile sampling.py` và `%%writefile features.py` với **nội dung đọc trực tiếp từ `realtime/shared/`** để luôn đồng bộ; các ô sau là load dữ liệu (Add Input → Notebook Output của `01_extract_pose`), dựng instance, train, đánh giá, export. Người dùng sẽ upload file `.ipynb` này lên Kaggle (File → Import Notebook) và chạy trên GPU T4.

### 7.4 Realtime (`detectors/fight.py`)
- Mỗi track giữ bộ đệm 32 frame mẫu gần nhất (deque). Mỗi 4 tick lấy mẫu: dựng các cặp ứng viên bằng **cùng hàm** trong `shared/features.py`, gom thành 1 batch, chạy ONNX Runtime (`CPUExecutionProvider`), sigmoid.
- Làm mượt mỗi cặp bằng EMA (α = 0.4). Báo **ĐÁNH NHAU** khi EMA > `threshold` liên tục ≥ 1 s. Trả `highlight_ids` = 2 ID của cặp → vẽ khung đỏ.
- **Khi chưa có `pair_tcn.onnx`:** dùng luật giả (2 người khoảng cách tâm < 1 × chiều cao và tốc độ cổ tay trung bình > 2 chiều cao/giây trong 1 s) để chạy thử toàn hệ thống. Tự chuyển sang ONNX khi file tồn tại.

---

## 8. Danh sách việc cần làm (theo thứ tự)

### Giai đoạn A – Hệ thống trên máy (làm ngay, không cần chờ Kaggle)
- [x] **A0.** Nếu các file bộ khung tuần 1 (`realtime/main.py`, `realtime/config.yaml`, `realtime/core/camera.py`, `realtime/core/pose.py`, `realtime/core/ui.py`, `realtime/shared/sampling.py`, các `__init__.py`) **chưa có**, tạo chúng đúng nguyên văn theo **mục 11**. Sau đó chạy `python realtime/main.py`, xác nhận: khung xương + ID ổn định, FPS ~25–30, tiếng Việt có dấu đúng, phím `s` lưu ảnh. Sửa lỗi nếu có.
- [x] **A1.** `detectors/base.py` + `core/events.py` (mục 5.3, 5.4). Mở rộng `config.yaml` thêm mục `detectors:` chứa bật/tắt và ngưỡng của từng detector.
- [x] **A2.** `core/db.py` (mục 5.5). `main.py`: mở session khi chạy, đóng khi thoát; mỗi sự kiện lưu ảnh chụp (tên file không dấu) + ghi DB + thêm vào bảng cảnh báo.
- [x] **A3.** `detectors/fall.py`, `detectors/pose_rules.py`, `detectors/panic.py` theo **code tham chiếu ở mục 9** (chuyển sang giao diện `Detector`). Thêm tham số vào `config.yaml`.
- [x] **A4.** `detectors/fight.py` bản luật giả (mục 7.4). Ghép tất cả vào `main.py` qua `EventManager`; người trong `highlight_ids` vẽ đỏ.
- [x] **A5.** `core/recorder.py`: giữ 5 s khung hình đã vẽ gần nhất (ring buffer); khi có sự kiện ghi clip `.mp4` (3 s trước + 2 s sau, codec `mp4v`) ở luồng nền, xong gọi `db.set_clip`.
- [x] **A6.** `tools/review.py`: duyệt sự kiện chưa review (phím y = đúng, n = báo nhầm, s = bỏ qua, q = thoát), in bảng `db.stats()`.
- **Mốc A:** giơ tay, té, nằm im trước webcam đều được báo đúng, có ảnh + clip + bản ghi DB, không báo trùng lặp.
  → Giơ tay, vẫy tay, ảnh + clip + DB đã đạt khi thử thật. **Còn thiếu: thử té ngã / nằm im toàn thân.**

### Giai đoạn B – Kaggle (khi `01_extract_pose` chạy xong)
- [x] **B1.** `shared/features.py` (mục 7.1) + test nhỏ bằng dữ liệu giả.
- [x] **B2.** `tools/build_kaggle_notebook.py` → `kaggle/02_train.ipynb` (mục 7.2, 7.3).
- [x] **B3.** Hướng dẫn người dùng chạy trên Kaggle, đọc kết quả, tải `pair_tcn.onnx` + `fight_config.json` về `realtime/models/`.
- **Mốc B:** F1 val ≥ 0.8. → **ĐẠT:** gộp val F1 0.850, AUC 0.883 (mức tiêu chí 7).

### Giai đoạn C – Ghép mô hình thật
- [x] **C1.** `fight.py` dùng ONNX (mục 7.4). Kiểm tra cùng một clip cho ra cùng điểm giữa Kaggle và local (sai khác do imgsz 640 vs 416 là chấp nhận được). *(Phần dùng ONNX đã xong; phần so sánh điểm Kaggle–local cho cùng clip CHƯA làm.)*
- **Mốc C (MVP):** giả vờ đánh nhau trước webcam → báo đúng cặp, khung đỏ.
  → Trên video RWF val đúng 5/6 clip. **Còn thiếu: thử webcam với 2 người thật.**

### Giai đoạn D – Hoàn thiện, đánh giá
- [ ] **D1.** *(chưa làm)* Hỗ trợ fine-tune bằng video tự quay (`data_selfrecorded/`): đánh nhau dàn dựng nhẹ + mẫu âm khó (ôm nhau, đập tay, vươn vai, vẫy chào, cúi xem điện thoại). Tạo script trích xuất khung xương chạy được trên Kaggle cho thư mục này.
- [x] **D2.** `tools/evaluate.py`: chạy pipeline không hiển thị trên thư mục video + file nhãn CSV, xuất precision/recall từng loại, FPS, độ trễ báo động. Dataset kiểm tra luật: **UR Fall / Le2i** (té ngã), **UMN** (hoảng loạn). *(Đã có công cụ – nhãn lấy theo tiền tố tên file thay cho file CSV; CHƯA chạy trên UR Fall / Le2i / UMN.)*
- [x] **D3.** `run.bat` (kích hoạt venv + chạy main), `README.md` hướng dẫn cài đặt/chạy.

### Giai đoạn E – LUỒNG 2: nhìn cả cảnh (mục 12)
- [x] **E0.** `shared/scene.py`, mục `scene` trong `config.yaml`, `tools/build_scene_notebooks.py`, test `tests/test_scene.py`.
- [ ] **E1.** `kaggle/04_scene_features.ipynb` – *đã viết + chạy thử trên máy; CHỜ chạy QUICK rồi đầy đủ trên Kaggle.*
- [ ] **E2.** `kaggle/05_scene_train.ipynb` – train MIL (ranking loss Sultani 2018) + head phân nhóm; đánh giá; xuất ONNX.
- [ ] **E3.** `detectors/scene.py` – luồng nền 2 FPS (backbone OpenVINO), làm mượt, báo ≥ 2 s; đo FPS luồng 1 không giảm.
- [ ] **E4.** DB thêm `scene_score`, `scene_group`, `fight_score` cho mọi sự kiện; `evaluate.py` bảng so sánh
  chỉ luồng 1 / chỉ luồng 2 / kết hợp.
- **Mốc E:** AUC mức frame trên UCF test ≥ 0.75 (tham khảo: Sultani 2018 với C3D ≈ 0.75); FPS luồng 1 không giảm.
- **Việc sau (chưa làm):** cháy/khói bằng YOLO riêng.

**Nếu thiếu thời gian, cắt theo thứ tự:** ÔM ĐẦU / GỤC ĐẦU → demo HOẢNG LOẠN chỉ bằng video UMN. **Luôn giữ:** đánh nhau, té ngã, giơ tay đầu hàng, database.

---

## 9. Code tham chiếu cho các luật (đã thiết kế – chuyển sang giao diện `Detector`)

### 9.1 Té ngã
```python
import numpy as np
from collections import deque

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12

class FallDetector:
    def __init__(self, fps=15, drop_speed=1.2, angle_deg=60, hold_sec=2.0):
        self.fps, self.drop_speed = fps, drop_speed       # drop_speed: số "thân người" rơi trong 1s
        self.angle, self.hold = angle_deg, int(hold_sec * fps)
        self.hist, self.dropped, self.down_count = {}, {}, {}

    def update(self, tid, kpts, box):
        if min(kpts[[L_SH, R_SH, L_HIP, R_HIP], 2]) < 0.4:
            return False
        sh  = kpts[[L_SH, R_SH], :2].mean(0)
        hip = kpts[[L_HIP, R_HIP], :2].mean(0)
        dx, dy = hip - sh
        angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))   # 0° = thân thẳng đứng
        w, h = box[2] - box[0], box[3] - box[1]
        q = self.hist.setdefault(tid, deque(maxlen=self.fps))
        q.append((hip[1], max(w, h)))
        if len(q) == q.maxlen:
            body = np.median([b for _, b in q])
            speed = (q[-1][0] - q[0][0]) / body                   # >0: hông đi xuống
            if speed > self.drop_speed:
                self.dropped[tid] = True
        lying = angle > self.angle or w > 1.2 * h
        if self.dropped.get(tid) and lying:
            self.down_count[tid] = self.down_count.get(tid, 0) + 1
        elif not lying:
            self.dropped[tid], self.down_count[tid] = False, 0
        return self.down_count.get(tid, 0) >= self.hold
```
(Cần thêm: chỉ báo 1 lần mỗi lần té, xoá dữ liệu ID đã rời khung hình.)

### 9.2 Luật tư thế: GIƠ TAY ĐẦU HÀNG, VẪY TAY CẦU CỨU, NẰM BẤT ĐỘNG, ÔM ĐẦU / GỤC ĐẦU
Điều kiện: giơ tay = cả 2 cổ tay cao hơn đầu ≥ 2.5 s; vẫy tay = cổ tay cao hơn vai, đổi hướng ≥ 5 lần, biên độ ≥ 0.4 thân trong 3 s; nằm bất động = thân ngang (góc > 60° hoặc w > 1.2h) và khớp gần như không di chuyển (< 0.08 thân/giây) ≥ 10 s; ôm đầu/gục = 2 cổ tay cách đầu < 0.5 thân hoặc đầu thấp hơn vai ≥ 10 s. Mọi khoảng cách chuẩn hoá theo chiều dài thân (vai→hông; nếu không thấy hông dùng 1.2 × độ rộng vai). Cooldown 15 s mỗi (ID, luật).

```python
import numpy as np
from collections import deque

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
HEAD_KPTS = [0, 1, 2, 3, 4]
CONF = 0.4

def ok(k, *idx):
    return all(k[i, 2] >= CONF for i in idx)

class PoseRules:
    def __init__(self, fps=15, hands_up_sec=2.5, wave_sec=3.0, wave_turns=5,
                 still_sec=10.0, distress_sec=10.0, cooldown_sec=15.0):
        self.fps = fps
        self.need = {"GIƠ TAY ĐẦU HÀNG": int(hands_up_sec * fps),
                     "NẰM BẤT ĐỘNG":     int(still_sec * fps),
                     "ÔM ĐẦU / GỤC ĐẦU": int(distress_sec * fps)}
        self.wave_len, self.wave_turns = int(wave_sec * fps), wave_turns
        self.cooldown = int(cooldown_sec * fps)
        self.hist, self.count, self.cool = {}, {}, {}

    def _scale(self, k):
        if not ok(k, L_SH, R_SH):
            return None
        sh = k[[L_SH, R_SH], :2].mean(0)
        if ok(k, L_HIP, R_HIP):
            torso = np.linalg.norm(k[[L_HIP, R_HIP], :2].mean(0) - sh)
        else:
            torso = 1.2 * np.linalg.norm(k[L_SH, :2] - k[R_SH, :2])
        return (sh, torso) if torso > 1 else None

    def _head(self, k):
        vis = [i for i in HEAD_KPTS if k[i, 2] >= CONF]
        return k[vis, :2].mean(0) if vis else None

    def _hands_up(self, k, sh, torso):
        if not ok(k, L_WR, R_WR):
            return False
        head = self._head(k)
        top = head[1] if head is not None else sh[1] - 0.5 * torso
        return k[L_WR, 1] < top - 0.1 * torso and k[R_WR, 1] < top - 0.1 * torso

    def _waving(self, q, torso):
        if len(q) < self.wave_len:
            return False
        arr = np.array(list(q)[-self.wave_len:])
        for wr, s in ((L_WR, L_SH), (R_WR, R_SH)):
            good = (arr[:, wr, 2] >= CONF) & (arr[:, s, 2] >= CONF)
            if good.mean() < 0.8:
                continue
            above = (arr[good, wr, 1] < arr[good, s, 1]).mean() > 0.8
            x = arr[good, wr, 0]
            if not above or (x.max() - x.min()) < 0.4 * torso:
                continue
            v = np.diff(x)
            v = v[np.abs(v) > 0.02 * torso]
            turns = np.sum(np.sign(v[1:]) != np.sign(v[:-1]))
            if turns >= self.wave_turns:
                return True
        return False

    def _lying_still(self, k, q, sh, torso, box):
        if not ok(k, L_HIP, R_HIP) or len(q) <= self.fps:
            return False
        dx, dy = k[[L_HIP, R_HIP], :2].mean(0) - sh
        angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
        w, h = box[2] - box[0], box[3] - box[1]
        if not (angle > 60 or w > 1.2 * h):
            return False
        a, b = q[-self.fps - 1], q[-1]
        m = (a[:, 2] >= CONF) & (b[:, 2] >= CONF)
        if m.sum() < 5:
            return False
        move = np.linalg.norm(b[m, :2] - a[m, :2], axis=1).mean() / torso
        return move < 0.08

    def _distress(self, k, sh, torso):
        head = self._head(k)
        if head is None:
            return False
        holding = ok(k, L_WR, R_WR) and max(
            np.linalg.norm(k[L_WR, :2] - head),
            np.linalg.norm(k[R_WR, :2] - head)) < 0.5 * torso
        slumped = head[1] > sh[1] + 0.1 * torso
        return holding or slumped

    def _fire(self, key, cond):
        if self.cool.get(key, 0) > 0:
            self.cool[key] -= 1
            return False
        if cond:
            self.cool[key] = self.cooldown
            return True
        return False

    def update(self, tid, kpts, box):
        q = self.hist.setdefault(tid, deque(maxlen=max(self.wave_len, self.fps) + 1))
        q.append(kpts.copy())
        s = self._scale(kpts)
        if s is None:
            return []
        sh, torso = s
        up = self._hands_up(kpts, sh, torso)
        conds = {
            "GIƠ TAY ĐẦU HÀNG": up,
            "NẰM BẤT ĐỘNG":     self._lying_still(kpts, q, sh, torso, box),
            "ÔM ĐẦU / GỤC ĐẦU": (not up) and self._distress(kpts, sh, torso),
        }
        events = []
        for rule, c in conds.items():
            key = (tid, rule)
            self.count[key] = self.count.get(key, 0) + 1 if c else 0
            if self._fire(key, self.count[key] >= self.need[rule]):
                events.append(rule)
        if self._fire((tid, "VẪY TAY CẦU CỨU"), self._waving(q, torso)):
            events.append("VẪY TAY CẦU CỨU")
        return events

    def prune(self, active_ids):
        active = set(active_ids)
        for tid in list(self.hist):
            if tid not in active:
                del self.hist[tid]
        self.count = {k: v for k, v in self.count.items() if k[0] in active}
        self.cool  = {k: v for k, v in self.cool.items()  if k[0] in active}
```
Lưu ý: "gục đầu" dễ nhầm với cúi xem điện thoại → để `distress_sec` cấu hình được, có tuỳ chọn tắt nhánh `slumped`.

### 9.3 Hoảng loạn (đám đông bỏ chạy)
Điều kiện (cùng lúc, kéo dài ≥ 1 s): ≥ `min_people` người (mặc định 4, demo trong phòng 3); ≥ 3 người **và** ≥ 50% đang chạy (tốc độ > 1.5 chiều cao người/giây); tốc độ trung bình cả cảnh > 2 × mức nền (mức nền học dần khi cảnh bình thường, cần ~10–20 s khởi động). Phân loại kiểu: tỏa ra nhiều hướng / chạy cùng một hướng / chạy hỗn loạn. Cooldown 10 s. Phải gọi mọi tick, kể cả khi không có ai.

```python
import numpy as np
from collections import deque

class PanicDetector:
    def __init__(self, fps=15, run_speed=1.5, min_people=4, min_running=3,
                 run_ratio=0.5, surge_ratio=2.0, hold_sec=1.0, cooldown_sec=10):
        self.fps = fps
        self.run_speed = run_speed
        self.min_people, self.min_running = min_people, min_running
        self.run_ratio, self.surge_ratio = run_ratio, surge_ratio
        self.hold, self.cooldown = int(hold_sec * fps), int(cooldown_sec * fps)
        self.tracks = {}
        self.baseline = None
        self.on_count, self.cool = 0, 0

    def update(self, ids, boxes, running_ids=()):
        seen = set()
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            q = self.tracks.setdefault(tid, deque(maxlen=self.fps))
            foot, h = ((x1 + x2) / 2, y2), (y2 - y1)
            if q and np.hypot(foot[0] - q[-1][0], foot[1] - q[-1][1]) > 1.5 * h:
                q.clear()
            q.append((foot[0], foot[1], h))
            seen.add(tid)
        for tid in list(self.tracks):
            if tid not in seen:
                del self.tracks[tid]

        speeds, vels, starts = {}, {}, {}
        for tid, q in self.tracks.items():
            if len(q) < q.maxlen:
                continue
            arr = np.array(q)
            dt = (len(arr) - 1) / self.fps
            body = np.median(arr[:, 2])
            d = arr[-1, :2] - arr[0, :2]
            speeds[tid] = np.linalg.norm(d) / body / dt
            vels[tid], starts[tid] = d, arr[0, :2]

        if self.cool > 0:
            self.cool -= 1
        n = len(speeds)
        if n == 0:
            self.on_count = 0
            return None

        mean_speed = float(np.mean(list(speeds.values())))
        running = [t for t, s in speeds.items() if s > self.run_speed or t in running_ids]
        if self.baseline is None:
            self.baseline = mean_speed
        surge = mean_speed > self.surge_ratio * max(self.baseline, 0.2)
        panic_now = (n >= self.min_people and len(running) >= self.min_running
                     and len(running) / n >= self.run_ratio and surge)
        if not panic_now:
            self.baseline = 0.99 * self.baseline + 0.01 * mean_speed
        self.on_count = self.on_count + 1 if panic_now else 0
        if self.on_count < self.hold or self.cool > 0:
            return None
        self.cool = self.cooldown

        center = np.mean([starts[t] for t in running], axis=0)
        outward = []
        for t in running:
            r, v = starts[t] - center, vels[t]
            if np.linalg.norm(r) > 1e-6 and np.linalg.norm(v) > 1e-6:
                outward.append(np.dot(r, v) / (np.linalg.norm(r) * np.linalg.norm(v)))
        dirs = np.array([vels[t] / (np.linalg.norm(vels[t]) + 1e-6) for t in running])
        if outward and np.mean(outward) > 0.5:
            kind = "tỏa ra nhiều hướng"
        elif np.linalg.norm(dirs.mean(0)) > 0.7:
            kind = "chạy cùng một hướng"
        else:
            kind = "chạy hỗn loạn"
        return dict(n_people=n, n_running=len(running), kind=kind, ids=running)
```

---

## 10. Quy ước & lưu ý

- Đường dẫn file ghi ra (ảnh, clip) dùng **tên không dấu** (hàm `slug` bỏ dấu tiếng Việt). Chữ hiển thị trên giao diện thì có dấu, vẽ bằng Pillow (`core/ui.py`) vì `cv2.putText` không hỗ trợ dấu.
- Tên cửa sổ OpenCV để không dấu.
- Tham số/ngưỡng đặt trong `realtime/config.yaml`, không hard-code.
- Không dùng `uvicorn --reload` / web server (đã chốt chạy cửa sổ local).
- Không đẩy file nặng lên git (đã có `.gitignore`). Commit sau mỗi bước hoàn thành, message tiếng Việt ngắn gọn.
- Notebook Kaggle: người dùng tự chạy trên Kaggle; Claude Code chỉ chuẩn bị file `.ipynb` và hướng dẫn từng bước bấm gì.
- Khi cần thông tin không có trong file này (đường dẫn, kết quả chạy, lỗi), **hỏi người dùng** thay vì đoán.

---

## 11. Code bộ khung tuần 1 (tạo nguyên văn nếu chưa có)
Ba file rỗng: `realtime/core/__init__.py`, `realtime/detectors/__init__.py`, `realtime/shared/__init__.py`.

### `realtime/config.yaml`
```yaml
# 0 = webcam laptop | "http://<ip-dien-thoai>:8080/video" | duong dan file video
source: 0

pose:
  model: models/yolov8n-pose_openvino_model
  imgsz: 416
  conf: 0.3

sample_fps: 15        # FPS cua chuoi khung xuong dua vao mo hinh (khop voi luc train)

ui:
  window: "Giam sat hanh vi"
  panel_width: 340
```

### `realtime/shared/sampling.py`
```python
class FpsSampler:
    """Chon frame theo thoi gian de chuoi dua vao mo hinh luon dung target_fps,
    bat ke camera/video goc chay 22, 25 hay 30 FPS."""

    def __init__(self, target_fps=15):
        self.period = 1.0 / target_fps
        self.next_t = None

    def tick(self, t):
        """t: thoi diem cua frame (giay). Tra ve True neu frame nay duoc lay mau."""
        if self.next_t is None:
            self.next_t = t + self.period
            return True
        if t + 1e-6 >= self.next_t:
            self.next_t += self.period
            if t - self.next_t > self.period:       # may bi lag qua xa -> bat kip, khong don frame
                self.next_t = t + self.period
            return True
        return False
```

### `realtime/core/camera.py`
```python
import threading, time
from pathlib import Path
import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


class VideoSource:
    """- Webcam / camera dien thoai: doc o luong rieng, luon tra frame moi nhat, tu ket noi lai.
       - File video: doc tuan tu tung frame, thoi gian tinh theo video (dung cho kiem thu)."""

    def __init__(self, src):
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.src = src
        self.is_file = isinstance(src, str) and Path(src).suffix.lower() in VIDEO_EXT
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon video: {src}")

        if self.is_file:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if 1 < fps < 121 else 30.0
            self.idx = 0
        else:
            self.frame, self.t, self.seq = None, 0.0, 0
            self.lock = threading.Lock()
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        if not self.is_file:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        fails = 0
        while self.running:
            ok, f = self.cap.read()
            if ok:
                fails = 0
                with self.lock:
                    self.frame, self.t, self.seq = f, time.monotonic(), self.seq + 1
            else:
                fails += 1
                time.sleep(0.05)
                if fails >= 40:                       # ~2 giay khong co hinh -> mo lai
                    print("Mat tin hieu camera, dang ket noi lai...")
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = self._open()
                    fails = 0

    def read(self):
        """Tra ve (ok, frame, thoi_diem_giay, so_thu_tu_frame)."""
        if self.is_file:
            ok, f = self.cap.read()
            if not ok:
                return False, None, None, None
            t = self.idx / self.fps
            self.idx += 1
            return True, f, t, self.idx
        with self.lock:
            return self.frame is not None, self.frame, self.t, self.seq

    def release(self):
        if not self.is_file:
            self.running = False
            time.sleep(0.1)
        self.cap.release()
```

### `realtime/core/pose.py`
```python
from ultralytics import YOLO


class PoseTracker:
    """YOLOv8n-pose + ByteTrack. Tra ve dict {track_id: {"box": xyxy, "kpts": (17,3)}} theo pixel."""

    def __init__(self, model_path, imgsz=416, conf=0.3):
        self.model = YOLO(str(model_path), task="pose")
        self.imgsz, self.conf = imgsz, conf

    def __call__(self, frame):
        r = self.model.track(frame, imgsz=self.imgsz, conf=self.conf, persist=True,
                             tracker="bytetrack.yaml", device="cpu", verbose=False)[0]
        tracks = {}
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy()
            for tid, b, k in zip(ids, boxes, kpts):
                tracks[tid] = {"box": b, "kpts": k}
        return tracks
```

### `realtime/core/ui.py`
```python
import os, time
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def _load_font(size):
    for p in ["C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT, FONT_B = _load_font(16), _load_font(20)


def draw_tracks(img, tracks, highlight=frozenset(), kp_thr=0.3):
    """Ve khung, khung xuong, ID. Nguoi trong `highlight` ve mau do."""
    for tid, tr in tracks.items():
        color = (0, 0, 255) if tid in highlight else (0, 200, 0)
        x1, y1, x2, y2 = tr["box"].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = tr["kpts"]
        for a, b in EDGES:
            if k[a, 2] > kp_thr and k[b, 2] > kp_thr:
                cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), color, 2)
        for x, y, c in k:
            if c > kp_thr:
                cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)
        cv2.putText(img, f"ID {tid}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_alert_border(img, thickness=8):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 0, 255), thickness)
    return img


def draw_panel(h, width, events, fps, n_people):
    """Bang thong tin ben phai, ho tro tieng Viet co dau."""
    panel = Image.new("RGB", (width, h), (25, 25, 25))
    d = ImageDraw.Draw(panel)
    d.text((12, 10), "GIÁM SÁT HÀNH VI", font=FONT_B, fill=(255, 255, 255))
    d.text((12, 40), f"FPS: {fps:.1f}    Số người: {n_people}", font=FONT, fill=(150, 230, 150))
    d.text((12, 66), "Cảnh báo gần đây:", font=FONT, fill=(200, 200, 200))
    y = 92
    if not events:
        d.text((12, y), "  (chưa có)", font=FONT, fill=(120, 120, 120))
    for e in events:
        if y > h - 40:
            break
        color = (255, 80, 80) if time.time() - e["t"] < 5 else (200, 200, 200)
        d.text((12, y), f"{e['time']}  {e['type']}", font=FONT, fill=color)
        ids = ", ".join(map(str, e.get("ids", []))) or "-"
        d.text((12, y + 20), f"    người: {ids}", font=FONT, fill=(160, 160, 160))
        y += 46
    return cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)
```

### `realtime/main.py`
```python
import sys, time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.camera import VideoSource
from core.pose import PoseTracker
from core.ui import draw_tracks, draw_panel, draw_alert_border
from shared.sampling import FpsSampler


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    src = sys.argv[1] if len(sys.argv) > 1 else cfg["source"]   # python main.py <nguon> de ghi de
    cam = VideoSource(src)
    pose = PoseTracker(ROOT / cfg["pose"]["model"], cfg["pose"]["imgsz"], cfg["pose"]["conf"])
    sampler = FpsSampler(cfg["sample_fps"])

    events = deque(maxlen=12)          # tuan 2: EventManager se them su kien vao day
    win, panel_w = cfg["ui"]["window"], cfg["ui"]["panel_width"]
    (ROOT / "events").mkdir(exist_ok=True)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps, t_last, last_seq = 0.0, time.perf_counter(), None
    print("Dang chay. Bam q de thoat, s de chup man hinh.")

    while True:
        ok, frame, t, seq = cam.read()
        if cam.is_file and not ok:
            break                                    # het video
        if not ok or seq == last_seq:
            time.sleep(0.003)                        # chua co frame moi tu camera
            continue
        last_seq = seq

        tracks = pose(frame)

        if sampler.tick(t):
            pass        # tuan 2-3: dua `tracks` vao bo dem cua cac detector o day

        out = draw_tracks(frame.copy(), tracks)
        if events and time.time() - events[0]["t"] < 3:
            draw_alert_border(out)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now

        canvas = np.hstack([out, draw_panel(out.shape[0], panel_w, events, fps, len(tracks))])
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            p = ROOT / "events" / f"manual_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(p), canvas)
            print("Da luu", p)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

---

## 12. LUỒNG 2 – nhìn ảnh màu cả cảnh (UCF-Crime), chạy song song luồng 1

**Mục đích:** luồng 1 (khung xương) giữ nguyên. Luồng 2 nhìn cả khung hình màu để bắt bất thường mức cảnh mà tư thế
không thể hiện (cướp, cháy nổ, đột nhập/trộm, tai nạn...). Hai luồng độc lập, kết quả cùng vào `EventManager`.

### 12.1 Thiết kế (đã chốt với người dùng 2026-11-08)
- **Đầu vào:** frame lấy mẫu **2 FPS theo thời gian** (`FpsSampler`), resize **224×224** (ép cả khung, không cắt) –
  `shared/scene.py::prep`.
- **Đặc trưng:** **MobileNetV3-Large** pretrained ImageNet (`IMAGENET1K_V2`), **đóng băng**, 1 vector 960 chiều / frame
  (sau global average pooling). Chuẩn hoá ImageNet nằm **trong** mô hình (vào uint8 RGB `(B, 224, 224, 3)`).
- **Đoạn:** 16 frame mẫu (~8 s) → mạng nhỏ (MLP hoặc GRU) → **điểm bất thường 0..1** + **nhóm**.
- **Nhóm** (ánh xạ trong `config.yaml: scene.groups` – nguồn duy nhất; lớp không có trong dữ liệu tự bỏ qua):
  BẠO LỰC: Fighting, Assault, Abuse, Arrest, Shooting · CƯỚP: Robbery · CHÁY NỔ: Arson, Explosion ·
  ĐỘT NHẬP/TRỘM: Burglary, Stealing, Shoplifting, Vandalism · TAI NẠN: RoadAccidents ·
  BÌNH THƯỜNG: mọi thư mục có chữ "Normal" + NonFight (RWF-2000) + NonViolence (RLVS).
  `scene.eval_only`: Fight (RWF), Violence (RLVS) → trích đặc trưng nhưng **không train**, chỉ đo AUC mức clip.
- **Train:** MIL kiểu **Sultani 2018** – ranking loss giữa video bất thường và video bình thường (nhãn cấp video:
  điểm đoạn cao nhất của video bất thường > của video bình thường; + ràng buộc mượt theo thời gian và thưa) +
  **head phân nhóm** chỉ học trên video bất thường (nhãn nhóm cấp video).
- **Đánh giá:** AUC **mức frame** trên UCF test (theo chú thích thời gian); AUC **mức clip** trên RWF/RLVS val;
  độ chính xác phân nhóm trên video bất thường test.
- **Xuất:** `scene_backbone.onnx` (uint8 → 960), `scene_head.onnx`, `scene_config.json` (fps, size, segment, nhóm, ngưỡng...).
- **Realtime (`detectors/scene.py`):** chạy ở **luồng nền 2 FPS** (backbone qua **OpenVINO**), giữ 16 vector gần nhất,
  làm mượt điểm, báo `CẢNH BẤT THƯỜNG: <nhóm>` khi điểm vượt ngưỡng **liên tục ≥ 2 s**, ưu tiên **50**, `ids = []`.
  Không được làm giảm FPS luồng 1 (đo lại khi bật / tắt).
- **DB:** mọi sự kiện ghi thêm `scene_score`, `scene_group`, `fight_score` (mục 5.5). `evaluate.py`: bảng so sánh
  **chỉ luồng 1 / chỉ luồng 2 / kết hợp**.
- **Cháy/khói bằng YOLO riêng:** việc sau, CHƯA làm.

### 12.2 Dữ liệu (cập nhật 2026-11-10)
- **Nguồn UCF chính:** `minhajuddinmeraj/anomalydetectiondatasetucf` (Version 50, 49 GB) – UCF-Crime ĐẦY ĐỦ + file split
  chính thức. **Chỉ lấy video từ:** `Anomaly-Videos-Part-1..4` (chấp nhận `Part_2`), `Training-Normal-Videos-*`,
  `Testing_Normal_Videos*`, `Normal_Videos_for_Event_Recognition*`, và thư mục lớp lẻ (`Burglary`, `Explosion`,
  `Shooting`) nếu không trùng tên file.
- **LOẠI hẳn** (video biến đổi có thể sinh từ video test → rò rỉ, hoặc không phải video gốc): `All_Data`, mọi thư mục
  chứa `Augmented`, `FightingA`, `Extracted`, `Per_Frame`, `Model_Trained`; `/anomaly/` (bản sao của mission-ai).
  Thư mục đặc trưng (`Extracted*`, `Per_Frame*`, `Model_Trained*`) không quét vào – `shared/scene.py::skip_reason`, `no_scan`.
- **Dự phòng:** `alirakhmaev/ucf-crime-full`, `mission-ai/crimeucfdataset` (có thể Add Input cùng, nhưng bản đầy đủ luôn
  được ưu tiên).
- **Bỏ trùng theo TÊN FILE**, giữ theo thứ tự `shared/scene.py::priority`: bản đầy đủ trước bản dự phòng; trong cùng bản,
  thư mục chính trước thư mục lớp lẻ / `Normal_Videos_for_Event_Recognition`; bản chính trước bản sao (RLVS).
  Notebook in bảng: mỗi lớp lấy từ thư mục nào, số video giữ / bị loại vì trùng / vì thư mục cấm.
- **Chia train/test – protocol CHÍNH THỨC:** `Anomaly_Train.txt` + `Normal_Train.txt` = train, `Anomaly_Test.txt` = test
  (mọi bản sao, kể cả trong `UCF_Crimes-Train-Test-Split*`, được đối chiếu), đối chiếu file chú thích thời gian (có chú
  thích = test; video ở cả train và test → xếp test). **`Anomaly_Train_New.txt` không dùng** (chỉ báo có giống không).
  Video UCF ngoài mọi danh sách → `ngoai_ds`, không dùng (`USE_UNLISTED_UCF = False`). Kỳ vọng: ~800 Normal train,
  150 Normal test, 810 bất thường train, 140 bất thường test (notebook so và đánh dấu `!!` nếu lệch > 2%).
- **Mẫu bình thường bổ sung:** NonFight (RWF-2000) + NonViolence (RLVS) – giữ (notebook dừng nếu thiếu); in tỉ lệ bất
  thường : bình thường của tập train theo số video và số phút. RWF train/val theo thư mục; RLVS theo `meta.csv` của
  `01_extract_pose` (khớp ≥ 90%), không có thì 80/20 stratified `random_state=42`. `val` của RWF/RLVS gọi chung là `test`.
- Nhận lớp theo tên thư mục (chuẩn hoá: bỏ hoa thường, dấu câu) hoặc tiền tố tên file; mọi thư mục có chữ "Normal" → Normal.
- Chú thích thời gian: frame ở FPS gốc, `-1` = không có (`shared/scene.py::parse_temporal_annotation`, `frame_labels`).

### 12.3 Notebook 04 – `kaggle/04_scene_features.ipynb` (sinh bởi `tools/build_scene_notebooks.py`)
- Ô đầu chép nguyên văn `shared/sampling.py`, `shared/scene.py`, mục `scene` của config.
- Giới hạn **15 phút/video**; GPU (fp16) cho backbone, worker CPU giải mã (chỉ `retrieve` frame được lấy mẫu).
- **Chạy tiếp được:** mỗi video xong ghi `scene_feats/<key>.npy` (float16, N × 960, ghi file tạm rồi đổi tên) + 1 dòng
  `scene_feats/_index.csv`; chạy lại bỏ qua video đã xong; Add Input output lần trước → tự chép sang; dừng an toàn sau
  `MAX_HOURS = 11`.
- **Đặc trưng đầy đủ đã trích ở lần trước** (Add Input output của notebook 04): cùng key `nguồn__lớp__tên file` →
  bỏ qua (kể cả khi lần trước lấy video từ bản UCF khác) và được chép sang output mới.
- `QUICK = True`: 4 video / (lớp, split), **ưu tiên video CHƯA có đặc trưng**, 2 phút đầu mỗi video, ghi riêng vào
  `scene_feats_quick/` (không dùng để train, không lẫn vào bản đầy đủ). Ước tính thời gian in cho **phần video mới**. Sau lượt chạy, notebook **ước tính thời gian chạy đầy đủ**: khớp
  giải mã ≈ a + b × phút video, GPU ≈ giây/frame, hiệu chỉnh bằng thời gian thật của lượt QUICK (thử trên máy: lệch ~30%).
- Thời điểm frame mẫu thứ i **không lưu** mà tính lại: `sampling.sample_indices([j / fps for j in range(n_read)], 2)`.
- Đầu ra: `scene_feats/`, `scene_meta.csv` (key, nguồn, lớp, nhóm, train_use, split, đường dẫn, fps, n_read, n_feats...),
  `ucf_annotation.csv`, `scene_inventory.csv` (số video / phút theo lớp, nhóm, split).

### 12.4 Notebook 05 – `kaggle/05_scene_train.ipynb` (CHƯA có trong repo – yêu cầu đã chốt 2026-11-10)
- **Protocol chuẩn UCF-Crime** (train/test theo danh sách chính thức từ `scene_meta.csv` của 04); có Normal UCF trong
  train nên **bỏ chế độ chia 75/75**.
- **Bag kiểu Sultani:** mỗi video chia **32 đoạn** theo thời gian, đặc trưng đoạn = trung bình các frame trong đoạn
  (**không đệm 0**; video ngắn hơn 32 frame mẫu thì các đoạn dùng chung frame).
- **Giữ:** chẩn đoán **lối tắt nguồn dữ liệu** (mô hình phân biệt nguồn UCF / RWF / RLVS thay vì bất thường), báo
  **độ chính xác theo nhóm**.
- **2 thí nghiệm để so sánh:** (a) UCF + NonFight/NonViolence làm mẫu bình thường bổ sung; (b) **chỉ UCF**.
- Đánh giá: AUC mức frame trên UCF test (chú thích thời gian), AUC mức clip RWF/RLVS val, độ chính xác phân nhóm.
- Xuất `scene_backbone.onnx`, `scene_head.onnx`, `scene_config.json`.
