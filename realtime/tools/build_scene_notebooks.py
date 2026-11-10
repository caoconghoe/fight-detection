"""Sinh các notebook LUỒNG 2 (nhìn cả cảnh, UCF-Crime) – CLAUDE.md mục 12.

  kaggle/04_scene_features.ipynb – trích đặc trưng MobileNetV3-Large (2 FPS, 224 px) cho mọi video trong /kaggle/input
Ô đầu chép NGUYÊN VĂN realtime/shared/sampling.py, realtime/shared/scene.py và mục `scene` của realtime/config.yaml
-> Kaggle và realtime dùng chung một code tiền xử lý / ánh xạ lớp.
Chạy (từ thư mục httm):  python realtime/tools/build_scene_notebooks.py
"""
import json
import sys
import textwrap
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kaggle_notebook import ROOT, code, md, writefile  # noqa: E402

OUT_04 = ROOT / "kaggle" / "04_scene_features.ipynb"

# Backbone dùng chung cho notebook 04 (trích đặc trưng) và 05 (xuất scene_backbone.onnx).
BACKBONE_SRC = """
class SceneBackbone(nn.Module):
    \"\"\"MobileNetV3-Large pretrained ImageNet, ĐÓNG BĂNG. Vào: (B, 224, 224, 3) uint8 RGB (đầu ra scene.prep).
    Ra: (B, 960) – vector sau global average pooling. Chuẩn hoá ImageNet nằm TRONG mô hình.\"\"\"
    def __init__(self, weights):
        super().__init__()
        m = torchvision.models.mobilenet_v3_large(weights=weights)
        self.features, self.pool = m.features, m.avgpool
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = (x.permute(0, 3, 1, 2).float() - self.mean) / self.std
        return self.pool(self.features(x)).flatten(1)
"""


def scene_cfg_cell():
    cfg = yaml.safe_load((ROOT / "realtime" / "config.yaml").read_text(encoding="utf-8"))["scene"]
    return code("%%writefile scene_groups.json\n" + json.dumps(cfg, ensure_ascii=False, indent=1))


CELLS_04 = [
    md("""
    # 04 – Trích đặc trưng cảnh (LUỒNG 2): MobileNetV3-Large, 2 FPS, 224 px

    Sinh tự động bởi `realtime/tools/build_scene_notebooks.py` – **không sửa tay**, sửa ở repo rồi sinh lại.

    **Chuẩn bị – Add Input:** `minhajuddinmeraj/anomalydetectiondatasetucf` (UCF-Crime ĐẦY ĐỦ + file split chính thức –
    **ưu tiên**) · `daniazehra/rwf-2000` · `mohamedmustafa/real-life-violence-situations-dataset` (NonFight / NonViolence:
    **bắt buộc**, mẫu bình thường bổ sung) · output của `01_extract_pose` (RLVS dùng đúng split cũ) · (tuỳ chọn, dự phòng)
    `alirakhmaev/ucf-crime-full`, `mission-ai/crimeucfdataset` · (nếu đã trích lần trước) output của chính notebook này.
    Settings → **GPU T4** · **Internet on** (tải trọng số ImageNet của MobileNetV3).

    Notebook **tự quét mọi dataset trong /kaggle/input**: nhận lớp theo tên thư mục (hoặc tiền tố tên file), loại thư mục
    cấm (`All_Data`, `*Augmented*`, `FightingA*`, `Extracted*`, `Per_Frame*`, `Model_Trained*`, `/anomaly/`), bỏ trùng
    theo tên file (ưu tiên bản đầy đủ), chia train/test theo **danh sách chính thức** của UCF-Crime.

    **Chạy:** `QUICK = True` thử (vài video / lớp, 2 phút đầu mỗi video) → ổn thì `QUICK = False`,
    *Save Version → Save & Run All*. **Bị ngắt / quá giờ:** Add Input output của chính notebook này rồi chạy lại –
    video đã xong được chép sang, chỉ trích phần còn lại.

    **Đầu ra (/kaggle/working):** `scene_feats/<key>.npy` (QUICK ghi riêng vào `scene_feats_quick/`, không dùng để
    train) (float16, N × 960; frame mẫu thứ i ở thời điểm theo
    `sampling.sample_indices` trên FPS gốc), `scene_feats/_index.csv` (tiến độ), `scene_meta.csv`, `ucf_annotation.csv`,
    `scene_inventory.csv` (số video / phút theo lớp, nhóm).
    """),
    md("## 1. Code dùng chung với realtime"),
    writefile("sampling.py"),
    writefile("scene.py"),
    scene_cfg_cell(),
    md("## 2. Cấu hình"),
    code("""
    QUICK = True              # True: chạy thử. Ổn rồi đổi False để trích đầy đủ.
    QUICK_PER_CLASS = 4       # QUICK: số video / (lớp, split)
    QUICK_MAX_SEC = 120       # QUICK: chỉ 2 phút đầu mỗi video
    MAX_HOURS = 11.0          # dừng an toàn trước giới hạn 12 h của Kaggle (chạy lại = làm tiếp)
    BATCH = 128               # frame / lượt GPU
    SEED = 42

    import os, sys, glob, json, time, shutil, hashlib, importlib
    from collections import Counter
    from multiprocessing import Pool
    import numpy as np
    import pandas as pd
    import cv2
    import torch
    import torch.nn as nn
    import torchvision

    INPUT = os.environ.get("HTTM_INPUT", "/kaggle/input")
    WORK = os.environ.get("HTTM_WORK", "/kaggle/working")
    N_WORKERS = int(os.environ.get("HTTM_WORKERS", os.cpu_count() or 2))
    FEATS = os.path.join(WORK, "scene_feats_quick" if QUICK else "scene_feats")   # QUICK (2 phút/video) KHÔNG lẫn vào bản đầy đủ
    os.makedirs(FEATS, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    import sampling, scene as S
    importlib.reload(sampling); importlib.reload(S)
    CFG = json.load(open("scene_groups.json", encoding="utf-8"))
    MAX_SEC = QUICK_MAX_SEC if QUICK else S.MAX_SEC
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "torchvision", torchvision.__version__, "| device:", DEVICE,
          "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "KHONG CO GPU (Settings -> Accelerator)")
    print(f"QUICK = {QUICK} | {S.FPS} FPS, {S.SIZE} px, toi da {MAX_SEC / 60:.0f} phut/video | CPU workers = {N_WORKERS}")
    print(f"MAX_SEC khi chay day du = {S.MAX_SEC} s ({S.MAX_SEC // 60} phut) | dang dung = {MAX_SEC} s")
    print("Nhom:", {g: len(c) for g, c in CFG["groups"].items()}, "| chi danh gia:", CFG["eval_only"])
    """),
    md("""
    ## 3. Quét mọi dataset trong /kaggle/input
    Nhận lớp → **loại thư mục cấm** (`All_Data`, `*Augmented*`, `FightingA*`, `Extracted*`, `Per_Frame*`, `Model_Trained*`:
    video biến đổi có thể sinh từ video test → rò rỉ; `/anomaly/` của mission-ai) → **bỏ trùng theo tên file**.
    Khi trùng, giữ theo thứ tự: bản UCF đầy đủ `anomalydetectiondatasetucf` trước 2 bản cũ (dự phòng); trong cùng bản,
    thư mục chính (`Anomaly-Videos-Part-*`, `Training-Normal-Videos-*`, `Testing_Normal_Videos*`) trước thư mục lớp lẻ
    (`Burglary`, `Explosion`, `Shooting`) và `Normal_Videos_for_Event_Recognition*`.
    Cùng lần duyệt thu luôn các file split / chú thích / `meta.csv` / `_index.csv` (không glob lại cả 49 GB).
    """),
    code("""
    INDEX = S.class_index(CFG["groups"], CFG["eval_only"])
    TXT_NAMES = {"anomaly_train.txt", "anomaly_test.txt", "normal_train.txt", "anomaly_train_new.txt"}
    TXT, META_CSVS, PREV_FEATS, cands, info_rows, unknown = {}, [], [], [], [], []

    def rel_parts(p):
        return os.path.relpath(p, INPUT).replace("\\\\", "/").split("/")

    def dataset_and_top(parts):
        \"\"\"(tên dataset ngắn, thư mục cấp đầu dưới gốc dataset) – vd ('anomalydetectiondatasetucf', 'Anomaly-Videos-Part-1').\"\"\"
        if parts[0] in ("datasets", "notebooks", "competitions", "models") and len(parts) > 4:
            return parts[2], parts[3]
        return parts[0], (parts[1] if len(parts) > 2 else "")

    for root, dirs, files in os.walk(INPUT):
        for d in list(dirs):
            if d == "scene_feats":
                PREV_FEATS.append(os.path.join(root, d))
        dirs[:] = sorted(d for d in dirs if not S.no_scan(d) and not d.startswith("scene_feats"))
        for f in files:
            fl = f.lower()
            if fl in TXT_NAMES or fl.startswith("temporal_anomaly_annotation"):
                TXT.setdefault("temporal_anomaly_annotation" if fl.startswith("temporal") else fl, []).append(
                    os.path.join(root, f))
            elif fl == "meta.csv":
                META_CSVS.append(os.path.join(root, f))
            if os.path.splitext(f)[1].lower() not in S.VIDEO_EXT:
                continue
            p = os.path.join(root, f).replace("\\\\", "/")
            parts = rel_parts(p)
            rel = "/".join(parts)
            ds, top = dataset_and_top(parts)
            cls = S.detect_class(p, INDEX)
            why = S.skip_reason(rel)
            row = dict(dataset=ds, top=top, cls=cls or "?", status="giu")
            if why:
                row["status"] = "cam: " + why
            elif cls is None:
                row["status"] = "khong ro lop"
                unknown.append(rel)
            info_rows.append(row)
            if row["status"] != "giu":
                continue
            src = S.source_of(cls)
            # UCF: trùng theo TÊN FILE giữa mọi bản / mọi thư mục; RWF/RLVS: theo (lớp, tên)
            dk = (src, fl) if src == "ucf" else (src, cls, fl)
            cands.append((dk, S.priority(rel, top, INDEX), p, src, cls, len(info_rows) - 1))
    print(f"Thu thap: {len(info_rows)} file video, {sum(len(v) for v in TXT.values())} file split/chu thich, "
          f"{len(META_CSVS)} meta.csv, {len(PREV_FEATS)} thu muc scene_feats cu")

    cands.sort()
    rows, kept, conflict = [], {}, 0
    for dk, pri, p, src, cls, i in cands:
        if dk in kept:
            info_rows[i]["status"] = "trung ten file"
            conflict += kept[dk] != cls
            continue
        kept[dk] = cls
        c, g, train_use = INDEX[S.norm(cls)]
        rows.append(dict(key=S.video_key(src, c, p), source=src, cls=c, group=g, train_use=train_use, path=p,
                         dataset=info_rows[i]["dataset"], top=info_rows[i]["top"]))
    vids = pd.DataFrame(rows)
    info = pd.DataFrame(info_rows)
    assert len(vids), "Khong tim thay video nao trong /kaggle/input -> Add Input cac dataset video"
    assert vids.key.is_unique, "Trung key: " + str(vids.key[vids.key.duplicated()].head().tolist())
    print(f"-> GIU {len(vids)} video | " + " | ".join(f"{k}: {v}" for k, v in info.status.value_counts().items()))
    if conflict:
        print(f"!! {conflict} file trung ten nhung 2 ban xep vao 2 lop khac nhau (giu lop cua ban uu tien)")
    if unknown:
        print("Vi du khong nhan ra lop:", unknown[:5])

    print("== Moi lop lay tu thu muc nao (giu / bi loai vi trung ten / vi thu muc cam)")
    src_tab = (info.assign(trang_thai=info.status.str.replace(r"^cam: .*", "thu muc cam", regex=True))
                   .groupby(["cls", "dataset", "top", "trang_thai"]).size().unstack("trang_thai", fill_value=0))
    with pd.option_context("display.max_rows", 300, "display.width", 250):
        display(src_tab)
    print("== Thu muc cam: so video bi loai theo luat")
    display(info[info.status.str.startswith("cam")].groupby(["dataset", "status"]).size().rename("so video"))
    missing = [c for g, cl in CFG["groups"].items() for c in cl if c not in set(vids.cls)]
    print("Lop co trong config nhung KHONG co trong du lieu (tu bo qua):", missing)
    """),
    code("""
    def probe(path):
        cap = cv2.VideoCapture(path)
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h, ok = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.isOpened()
        cap.release()
        fps = fps if 1 <= fps <= 120 else 30.0
        return dict(ok=ok, fps=fps, n_frames=int(max(n, 0)), width=int(w), height=int(h), dur_s=max(n, 0) / fps)

    t0 = time.time()
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            pr = pool.map(probe, vids.path, chunksize=16)
    else:
        pr = [probe(p) for p in vids.path]
    vids = pd.concat([vids, pd.DataFrame(pr)], axis=1)
    print(f"Doc thong tin {len(vids)} video trong {time.time() - t0:.0f} s | khong mo duoc: {(~vids.ok).sum()}")
    vids = vids[vids.ok].reset_index(drop=True)
    display(vids.groupby(["source", "width", "height"]).size().rename("so video").reset_index().head(20))
    """),
    md("""
    ## 4. Chia train / test – protocol CHÍNH THỨC của UCF-Crime
    UCF: `Anomaly_Train.txt` = **train** (Normal UCF không có mp4 → không dùng `Normal_Train.txt`; Normal train = RWF NonFight + RLVS NonViolence), `Anomaly_Test.txt` = **test** (mọi bản sao của các file này,
    kể cả trong `UCF_Crimes-Train-Test-Split*`, được đối chiếu với nhau), đối chiếu với file chú thích thời gian (video có
    chú thích = test). `Anomaly_Train_New.txt` KHÔNG dùng (chỉ báo nó có giống `Anomaly_Train.txt` không).
    Video UCF không có trong danh sách nào → `ngoai_ds`, **không dùng** (trừ khi `USE_UNLISTED_UCF = True`).
    RWF: thư mục `train` / `val`. RLVS: theo `meta.csv` của `01_extract_pose`, không có thì 80/20 stratified
    `random_state=42`. `val` của RWF/RLVS gọi chung là **test**.
    """),
    code("""
    USE_UNLISTED_UCF = False      # True: video UCF ngoài danh sách chính thức được đưa vào train

    import re
    def read_keys(text):
        out = set()
        for ln in text.splitlines():
            tok = ln.split()
            if tok and tok[0].lower() not in ("name", "file"):
                out.add(S.key_of(tok[0]))
        return out

    def read_lists(name):
        fs = sorted(TXT.get(name, []))
        sets = [read_keys(open(f, encoding="utf-8", errors="ignore").read()) for f in fs]
        return fs, set().union(*sets) if sets else set(), all(x == sets[0] for x in sets)

    LISTS = {n: read_lists(n) for n in ("anomaly_train.txt", "normal_train.txt", "anomaly_test.txt", "anomaly_train_new.txt")}
    for n, (fs, names, same) in LISTS.items():
        rel = [os.path.relpath(f, INPUT) for f in fs]
        print(f"{n:24s}: {len(fs)} ban, {len(names):4d} video" + ("" if same else "  !! CAC BAN KHAC NHAU")
              + (f"  {rel[:1]}" if rel else ""))
    for n in ("normal_train.txt", "anomaly_test.txt"):
        for f in sorted(TXT.get(n, []))[:1]:
            print(n, "- 5 dong dau (ten goc):", open(f, encoding="utf-8", errors="ignore").read().splitlines()[:5])
    ann_paths = sorted(TXT.get("temporal_anomaly_annotation", []))
    ANN = {}
    for f in ann_paths:
        ANN.update(S.parse_temporal_annotation(open(f, encoding="utf-8", errors="ignore").read()))
    print(f"File chu thich thoi gian: {len(ann_paths)} ban, {len(ANN)} video")
    TRAIN_K = LISTS["anomaly_train.txt"][1]   # Normal UCF khong co mp4 -> khong dung Normal_Train.txt
    TEST_K = LISTS["anomaly_test.txt"][1] | {S.key_of(k) for k in ANN}
    print(f"Khoa chuan hoa (bo duoi, dau gach, hoa thuong): train {len(TRAIN_K)}, test {len(TEST_K)}")
    if TRAIN_K & TEST_K:
        print(f"!! {len(TRAIN_K & TEST_K)} khoa vua o train vua o test -> xep TEST")
    ann_not_test = {S.key_of(k) for k in ANN} - LISTS["anomaly_test.txt"][1]
    if LISTS["anomaly_test.txt"][1] and ann_not_test:
        print(f"!! {len(ann_not_test)} video co chu thich nhung khong co trong Anomaly_Test.txt")
    new = LISTS["anomaly_train_new.txt"][1]
    if new:
        print("Anomaly_Train_New.txt:", "GIONG HET Anomaly_Train.txt" if new == LISTS["anomaly_train.txt"][1]
              else "KHAC -> khong dung")

    AT_FILES = sorted(TXT.get("anomaly_train.txt", []))
    if AT_FILES:
        from collections import Counter
        raw = [ln.split()[0] for ln in open(AT_FILES[0], encoding="utf-8", errors="ignore").read().splitlines()
               if ln.split() and ln.split()[0].lower() not in ("name", "file")]
        have_keys = set(vids.path.map(S.key_of))
        n_dup = len(raw) - len(set(S.key_of(x) for x in raw))
        unmatched = [(r, S.key_of(r)) for r in raw if S.key_of(r) not in have_keys]
        cause = Counter("Normal (loai khoi bat thuong)" if "normal" in k else "khong co file trong du lieu"
                        for _, k in unmatched)
        print(f"Anomaly_Train.txt: {len(raw)} dong, {len(set(S.key_of(x) for x in raw))} khoa duy nhat "
              f"({n_dup} dong trung), khop file that: {len(raw) - len(unmatched)}, khong khop: {len(unmatched)}")
        print("  nguyen nhan:", dict(cause))
        print("  5 ten dau khong khop:", [r for r, _ in unmatched[:5]])
    vids["fname"] = vids.path.map(os.path.basename)
    fl = vids.fname.str.lower()
    vids["stem"] = vids.path.map(S.key_of)
    vids["is_aug"] = vids.fname.map(S.is_augmented) & (vids.source == "ucf")
    vids["split"] = "train"
    ucf = vids.source == "ucf"
    UCF_SPLIT = "!! TAM: 80/20 (KHONG co file split)"
    if TRAIN_K or TEST_K:
        vids.loc[ucf, "split"] = "ngoai_ds"
        vids.loc[ucf & vids.stem.isin(TRAIN_K), "split"] = "train"
        def _norm_path(p):
            return re.sub("[^a-z0-9]", "", p.lower())
        in_ad = vids.path.map(lambda p: "anomalydataset" in _norm_path(p)) & ucf
        n_normal_before = int((vids.cls[in_ad] == "Normal").sum())
        vids.loc[in_ad, "cls"] = "Normal"                 # Anomaly_Dataset = video binh thuong (quyet dinh 2026-11-10)
        vids.loc[in_ad, "group"] = CFG["normal_group"]
        vids.loc[in_ad, "train_use"] = True
        print(f"Anomaly_Dataset: {int(in_ad.sum())} video, {n_normal_before} da co ten Normal, "
              f"{int(in_ad.sum()) - n_normal_before} doi sang Normal")
        nrm_folder_test = vids.path.map(lambda p: any(k in _norm_path(p) for k in
                                        ("testingnormal", "anomalydataset", "normalvideosforeventrecognition")))
        vids.loc[ucf & (vids.cls == "Normal") & nrm_folder_test & ~vids.stem.isin(TRAIN_K), "split"] = "test"
        vids.loc[ucf & vids.stem.isin(TEST_K), "split"] = "test"
        if USE_UNLISTED_UCF:
            vids.loc[ucf & (vids.split == "ngoai_ds"), "split"] = "train"
        nrm_u = ucf & (vids.cls == "Normal")
        vids.loc[nrm_u, "split"] = "ngoai_ds"
        vids.loc[nrm_u & nrm_folder_test, "split"] = "test"
        UCF_SPLIT = "danh sach chinh thuc"
    else:
        h = vids.key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5)
        vids.loc[ucf & (h == 0), "split"] = "test"

    nf = vids[ucf & (vids.cls == "Normal")]
    print(f"Normal UCF tim thay: {len(nf)} video, theo thu muc: {nf.top.value_counts().to_dict()}")
    print(f"  Normal UCF o test (thu muc Testing_Normal hoac chu thich): {int((nf.split == 'test').sum())}")
    if len(nf):
        print("  vi du ten file Normal:", nf.fname.head(5).tolist())
    n_aug_all = int(vids.is_aug.sum())
    aug_base_test = int((vids.is_aug & vids.fname.map(S.base_of).isin(TEST_K)).sum())
    print(f"Video augment (hau to A/flip...): {n_aug_all}; ban goc cua chung nam trong test: {aug_base_test}")

    rwf = vids.source == "rwf"
    in_val = vids.path.str.lower().str.contains("/val/") | vids.path.str.lower().str.contains("/test/")
    vids.loc[rwf & in_val, "split"] = "test"

    rlvs = vids.source == "rlvs"
    RLVS_SPLIT = "khong co video RLVS"
    if rlvs.any():
        old = {}
        for mp in META_CSVS:
            m = pd.read_csv(mp)
            if {"file", "dataset", "split"} <= set(m.columns):
                for f, d, sp in zip(m.file, m.dataset, m.split):
                    parts = os.path.basename(str(f))[:-4].split("__")
                    if str(d).lower() == "rlvs" and len(parts) >= 4:
                        old[S.norm("__".join(parts[2:-1]))] = "test" if str(sp) == "val" else "train"
        stem = vids.fname.map(lambda f: S.norm(os.path.splitext(f)[0]))
        hit = rlvs & stem.isin(old)
        if old and hit.sum() >= 0.9 * rlvs.sum():
            vids.loc[hit, "split"] = stem[hit].map(old)
            RLVS_SPLIT = f"theo meta.csv cua 01_extract_pose ({hit.sum()}/{rlvs.sum()} khop)"
            miss = rlvs & ~hit
        else:
            miss = rlvs
            RLVS_SPLIT = f"!! chia lai 80/20 stratified random_state=42 (meta.csv cua 01 khop {hit.sum()}/{rlvs.sum()})"
        if miss.any():
            from sklearn.model_selection import train_test_split
            sub = vids[miss].sort_values("fname")
            _, te = train_test_split(sub.index, test_size=0.2, stratify=sub.cls, random_state=SEED)
            vids.loc[te, "split"] = "test"
    vids = vids[~vids.is_aug].reset_index(drop=True)
    ucf = vids.source == "ucf"
    print(f"Da bo {n_aug_all} video augment khoi train va test")
    vids["minutes"] = np.minimum(vids.dur_s, S.MAX_SEC) / 60
    vids["is_norm"] = vids.group == CFG["normal_group"]
    tm = (vids.split == "train") & vids.train_use
    tot = vids[tm].groupby("is_norm").minutes.sum()
    vids["train_w"] = 0.0
    for flag in (True, False):
        sel = tm & (vids.is_norm == flag)
        if sel.any() and tot.get(flag, 0) > 0:
            vids.loc[sel, "train_w"] = 0.5 * vids.loc[sel, "minutes"] / tot[flag]
    print(f"Can bang theo phut (train_w, moi nhom tong 0.5): binh thuong {tot.get(True, 0):.1f} phut, "
          f"bat thuong {tot.get(False, 0):.1f} phut -> lay mau ti le train_w")
    vids["dur_c"] = np.minimum(vids.dur_s, S.MAX_SEC)     # thoi luong sau gioi han 15 phut (dung cho moi bang)
    ab = vids[vids.group != CFG["normal_group"]]
    print(f"Nhom BAT THUONG (tat ca split): {len(ab)} video, tong {ab.dur_s.sum():.0f} s thuc te, "
          f"{ab.dur_c.sum():.0f} s sau gioi han")
    print("5 video bat thuong dai nhat (thoi luong that):")
    for _, r in ab.nlargest(5, "dur_s").iterrows():
        print(f"  {r.dur_s:8.1f} s  {os.path.relpath(r.path, INPUT)}")
    n_long = int((vids.dur_s > S.MAX_SEC).sum())
    print(f"Video dai hon {S.MAX_SEC // 60} phut: {n_long} (extract chi lay {MAX_SEC // 60 if not QUICK else QUICK_MAX_SEC} giay dau; "
          f"bang thong ke dung minutes = min(dur, {S.MAX_SEC // 60} phut))")
    print("UCF :", UCF_SPLIT)
    print("RWF : thu muc train / val")
    print("RLVS:", RLVS_SPLIT)

    have = set(fl)
    ann_rows = []
    for k, (c, sp) in ANN.items():
        (s1, e1), (s2, e2) = (list(sp) + [(-1, -1), (-1, -1)])[:2]
        ann_rows.append(dict(fname=k, cls=c, s1=s1, e1=e1, s2=s2, e2=e2, co_trong_du_lieu=k.lower() in have))
    pd.DataFrame(ann_rows, columns=["fname", "cls", "s1", "e1", "s2", "e2", "co_trong_du_lieu"]).to_csv(
        os.path.join(WORK, "ucf_annotation.csv"), index=False)

    u = vids[vids.source == "ucf"]
    per = pd.crosstab(u.cls, u.split).reindex(columns=["train", "test", "ngoai_ds"], fill_value=0)
    per["phut_train"] = u[u.split == "train"].groupby("cls").minutes.sum().round(1)
    per["phut_test"] = u[u.split == "test"].groupby("cls").minutes.sum().round(1)
    per = per.fillna(0)
    have_keys = set(u.stem)
    print(f"Trong danh sach nhung KHONG co trong du lieu: train {len(TRAIN_K - have_keys)}, test {len(TEST_K - have_keys)}")
    print("== UCF: so video train / test theo lop (co trong du lieu) so voi danh sach chinh thuc")
    display(per)
    nrm = u.cls == "Normal"
    got = {"Normal test (ngoai mien)": (nrm & (u.split == "test")).sum(),
           "bat thuong train": (~nrm & (u.split == "train")).sum(), "bat thuong test": (~nrm & (u.split == "test")).sum()}
    expect = {"Normal test (ngoai mien)": 150, "bat thuong train": 810, "bat thuong test": 140}
    print("So voi ky vong:", " | ".join(f"{k}: {got[k]} (~{expect[k]})"
                                       + ("" if abs(got[k] - expect[k]) <= 0.02 * expect[k] else " !!") for k in expect))
    print("Normal train = RWF NonFight + RLVS NonViolence (ky vong ~1600, xem bang TAP TRAIN)")
    if (u.split == "ngoai_ds").any():
        print(f"Video UCF ngoai danh sach (khong dung): {(u.split == 'ngoai_ds').sum()} – vd {u.fname[u.split == 'ngoai_ds'].head(3).tolist()}")
    """),
    md("## 5. Thống kê: số video / số phút theo lớp và nhóm"),
    code("""
    vids["minutes"] = np.minimum(vids.dur_s, S.MAX_SEC) / 60
    inv = (vids.groupby(["group", "source", "cls", "split"]).agg(so_video=("key", "size"), so_phut=("minutes", "sum"))
               .unstack("split", fill_value=0))
    inv.columns = [f"{a}_{b}" for a, b in inv.columns]
    inv = inv.reset_index()
    print("== Theo lop (so phut da gioi han 15 phut/video)")
    display(inv.round(1))
    grp = (vids.assign(dung_de=np.where(vids.train_use, "train duoc", "chi danh gia"))
               .groupby(["group", "dung_de", "split"]).agg(so_video=("key", "size"), so_phut=("minutes", "sum")).round(1))
    print("== Theo nhom")
    display(grp)
    print(f"TONG: {len(vids)} video, {vids.minutes.sum() / 60:.1f} gio "
          f"(sau khi lay mau ~{vids.minutes.sum() * 60 * S.FPS / 1e6:.2f} trieu frame)")
    inv.to_csv(os.path.join(WORK, "scene_inventory.csv"), index=False)

    # Bình thường UCF chỉ ~200 video -> BẮT BUỘC thêm NonFight (RWF-2000) + NonViolence (RLVS) làm mẫu bình thường
    extra = [c for c in ("NonFight", "NonViolence") if not (vids.cls == c).any()]
    assert not extra, (f"Thieu mau binh thuong {extra} -> Add Input RWF-2000 (daniazehra/rwf-2000) va "
                       "RLVS (mohamedmustafa/real-life-violence-situations-dataset) roi chay lai")
    tr = vids[(vids.split == "train") & vids.train_use]
    is_norm = tr.group == CFG["normal_group"]
    ratio = (tr.assign(loai=np.where(is_norm, "binh thuong", "bat thuong"))
               .groupby(["loai", "source", "cls"]).agg(so_video=("key", "size"), so_phut=("minutes", "sum")).round(1))
    print("== TAP TRAIN: bat thuong / binh thuong (chi tinh lop dung de train)")
    display(ratio)
    nv, av = is_norm.sum(), (~is_norm).sum()
    nm, am = tr.minutes[is_norm].sum(), tr.minutes[~is_norm].sum()
    print(f"Ti le bat thuong : binh thuong = {av} : {nv} video ({av / max(nv, 1):.2f}) | "
          f"{am:.0f} : {nm:.0f} phut ({am / max(nm, 1e-9):.2f})")
    print(f"  binh thuong gom: UCF {(is_norm & (tr.source == 'ucf')).sum()} video, "
          f"RWF NonFight {(tr.cls == 'NonFight').sum()}, RLVS NonViolence {(tr.cls == 'NonViolence').sum()}")
    print("Do dai clip trung binh (giay) theo nhom va nguon:")
    display(tr.assign(loai=np.where(is_norm, "binh thuong", "bat thuong")).groupby(["loai", "source"])
              .dur_c.agg(["count", "mean", "min", "max"]).round(1))

    # Đặc trưng ĐẦY ĐỦ đã trích ở lần trước (Add Input output của notebook này): cùng key (= nguồn + lớp + tên file)
    # -> bỏ qua, kể cả khi lần trước lấy video từ bản UCF khác
    FULL_DONE = set()
    for d in PREV_FEATS + [os.path.join(WORK, "scene_feats")]:
        f_idx = os.path.join(d, "_index.csv")
        if os.path.exists(f_idx):
            old = pd.read_csv(f_idx)
            FULL_DONE |= {k for k in old.key[old.status == "ok"] if os.path.exists(os.path.join(d, k + ".npy"))}
    usable = vids[vids.split != "ngoai_ds"]
    have_prev = usable.key.isin(FULL_DONE)
    print(f"Dac trung day du da co tu lan truoc: {have_prev.sum()} / {len(usable)} video dung duoc "
          f"-> con MOI {(~have_prev).sum()} video, {usable.minutes[~have_prev].sum() / 60:.1f} gio")

    QUICK_MIN_NEW = 10
    q_idx = os.path.join(FEATS, "_index.csv")
    if QUICK:
        # QUICK phai trich THAT it nhat QUICK_MIN_NEW video de do thoi gian: uu tien video chua co cache QUICK,
        # roi xoa cache QUICK cua cac video duoc chon (cache QUICK khong dung cho ban day du)
        q_done = set(pd.read_csv(q_idx).key) if os.path.exists(q_idx) else set()
        rng = np.random.default_rng(SEED)
        order = usable.assign(_fresh=~usable.key.isin(q_done), _r=rng.random(len(usable)))
        order = order.sort_values(["_fresh", "_r"], ascending=[False, True])
        pick = []
        for _, g in order.groupby(["cls", "split"]):
            pick += list(g.index[:QUICK_PER_CLASS])
        pick = list(dict.fromkeys(pick))
        for i in order.index:
            if len(pick) >= QUICK_MIN_NEW:
                break
            if i not in pick:
                pick.append(i)
        plan = usable.loc[sorted(pick)].reset_index(drop=True)
        n_fresh = int((~plan.key.isin(q_done)).sum())
        for k in plan.key:
            fq = os.path.join(FEATS, k + ".npy")
            if os.path.exists(fq):
                os.remove(fq)
        if os.path.exists(q_idx):
            ix = pd.read_csv(q_idx)
            ix[~ix.key.isin(plan.key)].to_csv(q_idx, index=False)
        print(f"QUICK: chon {len(plan)} video ({n_fresh} chua co cache QUICK), da xoa cache QUICK cua chung de trich lai")
    else:
        plan = usable.reset_index(drop=True)
    print(f"Se trich {len(plan)} video" + (" (QUICK, ghi vao scene_feats_quick/)" if QUICK else ""))
    """),
    md("## 6. Backbone MobileNetV3-Large (đóng băng) – dùng chung với notebook 05"),
    code(BACKBONE_SRC + textwrap.dedent("""
    WEIGHTS = None if os.environ.get("HTTM_NO_PRETRAIN") else torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    backbone = SceneBackbone(WEIGHTS).to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        out = backbone(torch.zeros(2, S.SIZE, S.SIZE, 3, dtype=torch.uint8, device=DEVICE))
    assert out.shape == (2, S.FEAT_DIM), out.shape
    print("Backbone:", "IMAGENET1K_V2" if WEIGHTS else "!! KHONG co trong so (chi de chay thu)",
          f"| {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M tham so | ra {tuple(out.shape)}")
    """)),
    md("""
    ## 7. Trích đặc trưng (chạy tiếp được)
    Các worker CPU giải mã video (chỉ giải mã frame được lấy mẫu 2 FPS theo thời gian), GPU chạy backbone theo batch.
    Mỗi video xong ghi ngay `scene_feats/<key>.npy` + 1 dòng `_index.csv` → bị ngắt thì chạy lại ô này là làm tiếp.
    """),
    code("""
    INDEX_CSV = os.path.join(FEATS, "_index.csv")
    COLS = ["key", "status", "fps", "n_read", "n_feats", "bad_frames", "width", "height", "sec"]

    # nối tiếp từ output của lần chạy trước (Add Input output của chính notebook này) – chỉ khi trích ĐẦY ĐỦ
    for prev in ([] if QUICK else [os.path.join(d, "_index.csv") for d in PREV_FEATS]):
        if not os.path.exists(prev):
            continue
        old = pd.read_csv(prev)
        old = old[old.status == "ok"]
        n_copy = 0
        for k in old.key:
            src, dst = os.path.join(os.path.dirname(prev), k + ".npy"), os.path.join(FEATS, k + ".npy")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                n_copy += 1
        mine = pd.read_csv(INDEX_CSV) if os.path.exists(INDEX_CSV) else pd.DataFrame(columns=COLS)
        pd.concat([mine, old[~old.key.isin(mine.key)]]).to_csv(INDEX_CSV, index=False)
        print(f"Noi tiep tu {prev}: chep {n_copy} file")

    if not os.path.exists(INDEX_CSV):
        pd.DataFrame(columns=COLS).to_csv(INDEX_CSV, index=False)
    done = pd.read_csv(INDEX_CSV)
    done = set(done.key[(done.status == "ok") & done.key.map(lambda k: os.path.exists(os.path.join(FEATS, k + ".npy")))])
    todo = plan[~plan.key.isin(done)].reset_index(drop=True)
    print(f"Da xong {len(plan) - len(todo)} / {len(plan)} video | con {len(todo)}")


    class VideoFrames(torch.utils.data.Dataset):
        \"\"\"Giải mã 1 video -> frame lấy mẫu S.FPS theo thời gian (FpsSampler trên t = j / fps gốc), đã qua S.prep.\"\"\"
        def __init__(self, rows, max_sec):
            self.rows, self.max_sec = rows.to_dict("records"), max_sec

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            t0 = time.time()
            try:
                cap = cv2.VideoCapture(r["path"])
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if 1 <= fps <= 120 else 30.0
                smp, frames, last, bad, j = sampling.FpsSampler(S.FPS), [], None, 0, 0
                while j / fps < self.max_sec and cap.grab():
                    if smp.tick(j / fps):
                        ok, f = cap.retrieve()
                        if ok:
                            last = S.prep(f)
                        else:
                            bad += 1          # giữ đúng lưới thời gian: lặp frame trước (hoặc đen)
                        frames.append(last if last is not None else np.zeros((S.SIZE, S.SIZE, 3), np.uint8))
                    j += 1
                cap.release()
                arr = np.stack(frames) if frames else np.zeros((0, S.SIZE, S.SIZE, 3), np.uint8)
                return dict(key=r["key"], frames=arr, fps=fps, n_read=j, bad=bad, err="", sec=time.time() - t0)
            except Exception as e:                       # video hỏng: ghi lỗi, không dừng cả notebook
                return dict(key=r["key"], frames=np.zeros((0, S.SIZE, S.SIZE, 3), np.uint8), fps=0.0, n_read=0,
                            bad=0, err=f"{type(e).__name__}: {e}", sec=time.time() - t0)


    info = plan.set_index("key")[["width", "height"]].to_dict("index")
    dl = torch.utils.data.DataLoader(VideoFrames(todo, MAX_SEC), batch_size=None, shuffle=False,
                                     num_workers=N_WORKERS if N_WORKERS > 1 else 0,
                                     prefetch_factor=1 if N_WORKERS > 1 else None)
    STATS = []                                  # đo từng video (để ước tính thời gian chạy đầy đủ)
    t_start, last_print, n_ok, n_err, n_frames = time.time(), 0.0, 0, 0, 0
    with open(INDEX_CSV, "a", newline="") as fidx:
        for k_i, item in enumerate(dl):
            key = item["key"]
            frames = np.asarray(item["frames"])
            status = "ok"
            if item["err"]:
                status = "loi: " + str(item["err"])[:120].replace(",", ";")
            elif len(frames) == 0:
                status = "loi: khong doc duoc frame nao"
            if status == "ok":
                feats, t_gpu = [], time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                    for c in range(0, len(frames), BATCH):
                        x = torch.from_numpy(frames[c:c + BATCH]).to(DEVICE, non_blocking=True)
                        feats.append(backbone(x).float().cpu())
                f = torch.cat(feats).numpy().astype(np.float16)
                STATS.append(dict(key=key, minutes=int(item["n_read"]) / float(item["fps"]) / 60, frames=len(f),
                                  decode_s=float(item["sec"]), gpu_s=time.time() - t_gpu))
                tmp = os.path.join(FEATS, key + ".tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, f)
                os.replace(tmp, os.path.join(FEATS, key + ".npy"))      # ghi xong mới đổi tên -> không có file dở
                n_ok, n_frames = n_ok + 1, n_frames + len(f)
            else:
                n_err += 1
            wh = info.get(key, {})
            fidx.write(",".join(str(v) for v in [key, status, f"{float(item['fps']):.3f}", int(item["n_read"]),
                                                 len(frames) if status == "ok" else 0, int(item["bad"]),
                                                 wh.get("width", 0), wh.get("height", 0), f"{float(item['sec']):.1f}"]) + "\\n")
            fidx.flush()
            el = time.time() - t_start
            if el - last_print > 60 or k_i == len(todo) - 1:
                last_print = el
                rate = (k_i + 1) / el
                print(f"[{k_i + 1}/{len(todo)}] ok {n_ok}, loi {n_err} | {n_frames} frame | {el / 60:.1f} phut, "
                      f"con ~{(len(todo) - k_i - 1) / max(rate, 1e-9) / 60:.0f} phut", flush=True)
            if el > MAX_HOURS * 3600:
                print(f"!! Da chay {MAX_HOURS} h -> dung an toan. Save Version, Add Input output nay roi chay lai de lam tiep.")
                break
    del dl
    WALL = time.time() - t_start
    print(f"Xong luot nay: {n_ok} video ok, {n_err} loi, {n_frames} frame, {WALL / 60:.1f} phut")
    """),
    md("""
    ### Ước tính thời gian chạy ĐẦY ĐỦ (từ số đo của lượt vừa chạy – đọc kỹ khi `QUICK = True`)
    Giải mã mỗi video ≈ a + b × số phút (khớp tuyến tính trên các video vừa trích, worker CPU chạy song song);
    GPU ≈ (giây / frame) × số frame. Thời gian thật ≈ max(giải mã / số worker, GPU), hiệu chỉnh bằng tỉ lệ
    thời gian thật / dự đoán của chính lượt vừa chạy. Áp cho TOÀN BỘ video (giới hạn 15 phút/video).
    """),
    code("""
    st = pd.DataFrame(STATS)
    if len(st) < 5:
        print("Chua du so lieu de uoc tinh (can >= 5 video vua trich trong luot nay).")
    else:
        b, a = np.polyfit(st.minutes, st.decode_s, 1) if st.minutes.nunique() > 1 else (st.decode_s.sum() / st.minutes.sum(), 0)
        a, b = max(a, 0.0), max(b, 0.0)
        gpu_per_frame = st.gpu_s.sum() / max(st.frames.sum(), 1)
        workers = max(N_WORKERS, 1)

        def predict(minutes):
            m = np.asarray(minutes, float)
            return max((a + b * m).sum() / workers, gpu_per_frame * (m * 60 * S.FPS).sum())

        calib = WALL / max(predict(st.minutes), 1e-6)
        full_min = np.minimum(vids.dur_s, S.MAX_SEC) / 60
        # phần còn phải trích ĐẦY ĐỦ: video dùng được, chưa có đặc trưng đầy đủ từ lần trước (QUICK không tính là xong)
        finished = FULL_DONE if QUICK else FULL_DONE | done | set(st.key)
        rest = ((vids.split != "ngoai_ds") & ~vids.key.isin(finished)).values
        todo_min = full_min[rest]
        est = predict(todo_min) * calib
        print(f"Do tren {len(st)} video ({st.minutes.sum():.1f} phut video, {st.frames.sum()} frame), thuc te {WALL:.0f} s:")
        print(f"  giai ma: {a:.2f} s/video + {b:.2f} s/phut video (1 worker) | GPU: {gpu_per_frame * 1000:.2f} ms/frame "
              f"| hieu chinh x{calib:.2f}")
        print(f"  -> Con {len(todo_min)} video MOI can trich, {todo_min.sum() / 60:.1f} gio video ({todo_min.sum() * 60 * S.FPS / 1e6:.2f} trieu frame)")
        by = vids[rest].assign(m=todo_min).groupby("source").m
        for src, m in by:
            print(f"     {src:5s}: {len(m):5d} video, {m.sum() / 60:6.1f} gio video -> ~{predict(m) * calib / 60:7.1f} phut")
        print(f"  -> UOC TINH chay day du: ~{est / 60:.1f} phut ({est / 3600:.1f} gio)", end="")
        sessions = int(np.ceil(est / 3600 / MAX_HOURS))
        print(f" -> {'vua 1 phien Kaggle' if sessions <= 1 else f'can ~{sessions} phien (Add Input output de chay tiep)'}")
    """),
    md("""
    ## 7b. Đoạn cố định 32 (kiểu Sultani, KHÔNG đệm 0)
    Mỗi video (train/test, không tính augment) -> 32 đoạn theo thời gian, đặc trưng đoạn = trung bình các frame trong đoạn.
    Video dài hơn 32 frame: chia đều; video ngắn hơn 32 frame: lấy frame lặp theo lưới đều (không đệm 0).
    Ghi `scene_seg32/<key>.npy` (32 × 960, float16). Notebook 05 đọc các file này.
    """),
    code("""
    SEG_DIR = os.path.join(WORK, "scene_seg32_quick" if QUICK else "scene_seg32")
    os.makedirs(SEG_DIR, exist_ok=True)

    def seg32(f):
        n = len(f)
        if n >= 32:
            e = np.linspace(0, n, 33).astype(int)
            return np.stack([f[e[i]:max(e[i] + 1, e[i + 1])].astype(np.float32).mean(0) for i in range(32)])
        return f[np.linspace(0, n - 1, 32).round().astype(int)].astype(np.float32)

    n_seg, n_skip, n_bad = 0, 0, 0
    for k in vids[vids.split != "ngoai_ds"].key:
        src, dst = os.path.join(FEATS, k + ".npy"), os.path.join(SEG_DIR, k + ".npy")
        if not os.path.exists(src):
            n_skip += 1
            continue
        if os.path.exists(dst):
            continue
        g = seg32(np.load(src))
        if not np.isfinite(g).all() or (np.abs(g).sum(1) == 0).any():
            n_bad += 1
            print('!! doan 32 khong hop le (NaN hoac toan 0):', k)
        np.save(dst, g.astype(np.float16))
        n_seg += 1
    print(f"Doan 32: ghi {n_seg} video moi, bo qua {n_skip} video chua co dac trung (QUICK chi co mot phan), {n_bad} doan khong hop le")
    """),
    md("## 8. Tổng hợp `scene_meta.csv` + kiểm tra"),
    code("""
    idx = pd.read_csv(INDEX_CSV).drop_duplicates("key", keep="last")
    meta = plan.drop(columns=["fps", "n_frames", "width", "height", "ok"]).merge(idx, on="key", how="left")
    meta["status"] = meta.status.fillna("chua trich")
    ok = meta.status == "ok"
    print(f"Trich xong {ok.sum()}/{len(meta)} video | loi: {meta.status.str.startswith('loi').sum()} "
          f"| chua trich: {(meta.status == 'chua trich').sum()}")
    if (~ok).any():
        display(meta.loc[~ok, ["key", "status"]].head(10))
    meta["phut_da_trich"] = meta.n_feats.fillna(0) / S.FPS / 60
    display(meta[ok].groupby(["group", "cls", "split"]).agg(so_video=("key", "size"), so_phut=("phut_da_trich", "sum"),
                                                              frame_TB=("n_feats", "mean")).round(1))
    meta.to_csv(os.path.join(WORK, "scene_meta.csv"), index=False)

    # kiểm tra: số frame trong .npy khớp lưới lấy mẫu; giá trị hữu hạn; khác nhau giữa 2 video khác lớp
    rng = np.random.default_rng(SEED)
    for _, r in meta[ok].sample(min(5, ok.sum()), random_state=SEED).iterrows():
        f = np.load(os.path.join(FEATS, r.key + ".npy"))
        n_expect = len(sampling.sample_indices([j / r.fps for j in range(int(r.n_read))], S.FPS))
        assert f.shape == (n_expect, S.FEAT_DIM) and np.isfinite(f).all(), (r.key, f.shape, n_expect)
        print(f"  {r.key[:50]:50s} {f.shape} {f.dtype} | trung binh {float(f.mean()):.3f} | khop luoi 2 FPS: OK")
    size = sum(os.path.getsize(os.path.join(FEATS, f)) for f in os.listdir(FEATS)) / 1e9
    print(f"scene_feats/: {len(os.listdir(FEATS)) - 1} file, {size:.2f} GB")
    """),
]


def build(cells, out):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Da ghi {out} ({len(cells)} o)")


if __name__ == "__main__":
    build(CELLS_04, OUT_04)
