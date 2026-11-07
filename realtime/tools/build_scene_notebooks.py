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

    **Chuẩn bị:** Add Input các dataset video (UCF-Crime bản nào cũng được – có thể nhiều bản, RWF-2000, RLVS) ·
    (nên) Add Input output của `01_extract_pose` để RLVS dùng đúng split cũ · Settings → **GPU T4** · **Internet on**
    (tải trọng số ImageNet của MobileNetV3).

    Notebook **tự quét mọi dataset trong /kaggle/input**: nhận lớp theo tên thư mục (hoặc tiền tố tên file), bỏ mọi
    đường dẫn chứa `/anomaly/`, chống trùng theo (nguồn, lớp, tên file). Thêm bản UCF đầy đủ sau này → tự nhận thêm lớp.

    **Chạy:** `QUICK = True` thử (vài video / lớp, 2 phút đầu mỗi video) → ổn thì `QUICK = False`,
    *Save Version → Save & Run All*. **Bị ngắt / quá giờ:** Add Input output của chính notebook này rồi chạy lại –
    video đã xong được chép sang, chỉ trích phần còn lại.

    **Đầu ra (/kaggle/working):** `scene_feats/<key>.npy` (float16, N × 960; frame mẫu thứ i ở thời điểm theo
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
    FEATS = os.path.join(WORK, "scene_feats")
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
    print("Nhom:", {g: len(c) for g, c in CFG["groups"].items()}, "| chi danh gia:", CFG["eval_only"])
    """),
    md("""
    ## 3. Quét mọi dataset trong /kaggle/input
    Nhận lớp → bỏ `/anomaly/` → chống trùng theo (nguồn, lớp, tên file), ưu tiên bản chính (RLVS: bỏ bản sao
    `real life violence situations`). Lớp không có trong dữ liệu tự bỏ qua; lớp mới trong dataset thêm sau tự được nhận.
    """),
    code("""
    INDEX = S.class_index(CFG["groups"], CFG["eval_only"])
    cands, skipped, unknown = [], Counter(), []
    for root, dirs, files in os.walk(INPUT):
        if "scene_feats" in root.replace("\\\\", "/").split("/"):
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() not in S.VIDEO_EXT:
                continue
            p = os.path.join(root, f).replace("\\\\", "/")
            if S.skip_path(p):
                skipped["duong dan chua /anomaly/"] += 1
                continue
            cls = S.detect_class(p, INDEX)
            if cls is None:
                skipped["khong nhan ra lop"] += 1
                unknown.append(p)
                continue
            cands.append((S.source_of(cls), cls, f.lower(), S.dup_rank(p), p))
    cands.sort()
    rows, seen, dups = [], set(), Counter()
    for src, cls, fname, rank, p in cands:
        if (src, cls, fname) in seen:
            dups[src] += 1
            continue
        seen.add((src, cls, fname))
        c, g, train_use = INDEX[S.norm(cls)]
        rows.append(dict(key=S.video_key(src, c, p), source=src, cls=c, group=g, train_use=train_use, path=p))
    vids = pd.DataFrame(rows)
    assert len(vids), "Khong tim thay video nao trong /kaggle/input -> Add Input cac dataset video"
    assert vids.key.is_unique, "Trung key: " + str(vids.key[vids.key.duplicated()].head().tolist())
    print(f"Tim thay {len(cands)} video hop le, bo {sum(dups.values())} ban trung {dict(dups)}, bo khac: {dict(skipped)}")
    if unknown:
        print("Vi du khong nhan ra lop:", unknown[:5])
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
    ## 4. Chia train / test
    UCF: video có trong `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` (hoặc `Anomaly_Test.txt`) là **test**.
    RWF: thư mục `train` / `val` chính thức. RLVS: theo split của `01_extract_pose` (meta.csv, nếu đã Add Input),
    không có thì chia 80/20 stratified `random_state=42`. Ở đây `val` của RWF/RLVS gọi chung là **test**.
    """),
    code("""
    def read_any(pattern):
        fs = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
        return fs, (open(fs[0], encoding="utf-8", errors="ignore").read() if fs else "")

    ann_files, ann_text = read_any("Temporal_Anomaly_Annotation*.txt")
    list_files, list_text = read_any("Anomaly_Test.txt")
    ANN = S.parse_temporal_annotation(ann_text)
    UCF_TEST = set(ANN) | S.parse_test_list(list_text)
    print("File chu thich thoi gian:", ann_files[:1], f"({len(ANN)} video)")
    print("Danh sach test:", list_files[:1])
    vids["fname"] = vids.path.map(os.path.basename)
    vids["split"] = "train"
    ucf = vids.source == "ucf"
    if UCF_TEST:
        vids.loc[ucf & vids.fname.isin(UCF_TEST), "split"] = "test"
        UCF_SPLIT = "danh sach test chinh thuc"
    else:
        h = vids.key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5)
        vids.loc[ucf & (h == 0), "split"] = "test"
        UCF_SPLIT = "!! TAM: bam 80/20 (KHONG co file chu thich -> khong tinh duoc AUC muc frame)"
        print("!! KHONG tim thay Temporal_Anomaly_Annotation_for_Testing_Videos.txt / Anomaly_Test.txt trong /kaggle/input.")
        print("   Add Input mot dataset UCF-Crime co file nay (ban day du thuong co) roi chay lai.")

    rwf = vids.source == "rwf"
    in_val = vids.path.str.lower().str.contains("/val/") | vids.path.str.lower().str.contains("/test/")
    vids.loc[rwf & in_val, "split"] = "test"

    rlvs = vids.source == "rlvs"
    RLVS_SPLIT = "khong co video RLVS"
    if rlvs.any():
        old = {}
        for mp in glob.glob(os.path.join(INPUT, "**", "meta.csv"), recursive=True):
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
    print("UCF :", UCF_SPLIT)
    print("RWF : thu muc train / val")
    print("RLVS:", RLVS_SPLIT)

    have = set(vids.fname)
    ann_rows = []
    for k, (c, sp) in ANN.items():
        (s1, e1), (s2, e2) = (list(sp) + [(-1, -1), (-1, -1)])[:2]
        ann_rows.append(dict(fname=k, cls=c, s1=s1, e1=e1, s2=s2, e2=e2, co_trong_du_lieu=k in have))
    pd.DataFrame(ann_rows, columns=["fname", "cls", "s1", "e1", "s2", "e2", "co_trong_du_lieu"]).to_csv(
        os.path.join(WORK, "ucf_annotation.csv"), index=False)
    if ANN:
        t = vids[ucf & (vids.split == "test")]
        print(f"UCF test co trong du lieu: {len(t)} video ({t.cls.ne('Normal').sum()} bat thuong, "
              f"{t.cls.eq('Normal').sum()} binh thuong) / {len(ANN)} video trong file chu thich")
        if not t.cls.eq("Normal").any():
            print("  (chua co video Normal cua tap test -> AUC muc frame chi tinh tren cac video bat thuong)")
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

    if QUICK:
        plan = (vids.groupby(["cls", "split"], group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), QUICK_PER_CLASS), random_state=SEED)))
        plan = vids.loc[sorted(plan.index)].reset_index(drop=True)
    else:
        plan = vids
    print(f"Se trich {len(plan)} video" + (" (QUICK)" if QUICK else ""))
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

    # nối tiếp từ output của lần chạy trước (Add Input output của chính notebook này)
    for prev in glob.glob(os.path.join(INPUT, "**", "scene_feats", "_index.csv"), recursive=True):
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
                feats = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                    for c in range(0, len(frames), BATCH):
                        x = torch.from_numpy(frames[c:c + BATCH]).to(DEVICE, non_blocking=True)
                        feats.append(backbone(x).float().cpu())
                f = torch.cat(feats).numpy().astype(np.float16)
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
    print(f"Xong luot nay: {n_ok} video ok, {n_err} loi, {n_frames} frame, {(time.time() - t_start) / 60:.1f} phut")
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
