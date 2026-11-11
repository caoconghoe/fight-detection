"""Sinh kaggle/05_scene_train.ipynb – huấn luyện LUỒNG 2 trên CỬA SỔ 16 frame (CLAUDE.md mục 12.4).

Đọc output của notebook 04 (Add Input): scene_feats/<key>.npy (N × 960, 2 FPS), scene_meta.csv, ucf_annotation.csv.
Cửa sổ 16 frame (~8 s), bước 8 – đúng kích thước realtime chấm điểm. Ranking loss kiểu Sultani trên điểm cửa sổ,
head nhóm chỉ học trên cửa sổ bất thường. Xuất scene_head.onnx (input 16 × 960), scene_backbone.onnx, scene_config.json.
Chạy (từ thư mục httm):  python realtime/tools/build_scene_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kaggle_notebook import ROOT, code, md, writefile  # noqa: E402
from build_scene_notebooks import BACKBONE_SRC, build, scene_cfg_cell  # noqa: E402

OUT_05 = ROOT / "kaggle" / "05_scene_train.ipynb"

CELLS_05 = [
    md("""
    # 05 – Huấn luyện LUỒNG 2 trên cửa sổ 16 frame (khớp realtime)

    Sinh tự động bởi `realtime/tools/build_scene_train.py` – **không sửa tay**.

    **Chuẩn bị – Add Input:** output của notebook **04_scene_features** bản **đầy đủ** (có `scene_feats/`,
    `scene_meta.csv`, `ucf_annotation.csv`). Settings → **GPU T4** · **Internet on** (cài `onnx`, tải MobileNetV3 để xuất backbone).

    **Cửa sổ:** 16 frame mẫu 2 FPS (~8 s), bước 8. Video ngắn hơn 16 frame: lặp frame theo lưới đều, KHÔNG đệm 0,
    đánh dấu `short_clip` (chỉ để thống kê, không đưa vào mô hình).
    **Lấy mẫu:** chọn video theo `train_w` (cân bằng theo phút), rồi chọn cửa sổ trong video.

    **Chạy:** `QUICK = True` thử → ổn thì `QUICK = False` (Save & Run All).
    """),
    code("""
    import importlib.util, subprocess, sys
    need = [m for m in ("onnx", "onnxruntime") if importlib.util.find_spec(m) is None]
    if need:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript", "onnxruntime"])
        assert r.returncode == 0, "pip install that bai (" + ", ".join(need) + ") -> bat Internet: Settings -> Internet on"
    import onnx
    import onnxruntime
    print("onnx", onnx.__version__, "| onnxruntime", onnxruntime.__version__)
    """),
    md("## 1. Code dùng chung với realtime"),
    writefile("scene.py"),
    scene_cfg_cell(),
    md("## 2. Cấu hình"),
    code("""
    import os
    QUICK = True              # True: thu nhanh. Ok roi doi False de chay day du.
    QUICK_EPOCHS, QUICK_STEPS = 2, 20
    EPOCHS, STEPS = 20, 300
    BAG = 8                   # moi buoc: BAG video bat thuong + BAG video binh thuong
    BAGW = 8                  # moi video trong buoc: 8 cua so 16 frame (chuoi trai deu theo thoi gian)
    WIN, STRIDE = 16, 8
    LR, WD = 1e-3, 1e-4
    SEED = 42
    EXPORT_BACKBONE = True    # xuat scene_backbone.onnx (can Internet de tai trong so ImageNet)
    INPUT = os.environ.get("HTTM_INPUT", "/kaggle/input")
    WORK = os.environ.get("HTTM_WORK", "/kaggle/working")
    os.makedirs(WORK, exist_ok=True)

    import sys, glob, json, time, importlib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision
    from sklearn.metrics import roc_auc_score, f1_score
    sys.path.insert(0, os.getcwd())
    import scene as S
    importlib.reload(S)
    RETRAIN = False           # False: neu co scene_head.pt trong Add Input thi KHONG huan luyen lai, chi danh gia
    _pts = sorted(glob.glob(os.path.join(INPUT, "**", "scene_head.pt"), recursive=True))
    LOAD_PT = _pts[0] if (_pts and not RETRAIN) else None
    CFG = json.load(open("scene_groups.json", encoding="utf-8"))
    NORMAL = CFG["normal_group"]
    ABN_GROUPS = [g for g in CFG["groups"] if g != NORMAL]
    print("5 nhom bat thuong:", ABN_GROUPS)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| device:", DEVICE)
    print(f"QUICK = {QUICK} | MAX_SEC (15 phut khi chay day du) = {S.MAX_SEC} s | cua so {WIN} frame, buoc {STRIDE}")
    """),
    md("## 3. Dữ liệu từ notebook 04 (đặc trưng theo từng frame)"),
    code("""
    feat_dirs = sorted(d for d in glob.glob(os.path.join(INPUT, "**", "scene_feats*"), recursive=True) if os.path.isdir(d))
    assert feat_dirs, "Khong thay scene_feats/ -> Add Input output cua notebook 04_scene_features"
    full = [d for d in feat_dirs if os.path.basename(d) == "scene_feats"]
    FE_DIR = (full or feat_dirs)[0]
    print("Dac trung theo frame:", FE_DIR, "| tim thay:", [os.path.basename(d) for d in feat_dirs])
    META = os.path.join(os.path.dirname(FE_DIR), "scene_meta.csv")
    if not os.path.exists(META):
        META = glob.glob(os.path.join(INPUT, "**", "scene_meta.csv"), recursive=True)[0]
    ANN_F = (glob.glob(os.path.join(INPUT, "**", "ucf_annotation.csv"), recursive=True) or [None])[0]
    meta = pd.read_csv(META)
    meta = meta[meta.key.map(lambda k: os.path.exists(os.path.join(FE_DIR, k + ".npy")))].reset_index(drop=True)
    meta["ab"] = meta.group != NORMAL
    meta["gi"] = meta.group.map({g: i for i, g in enumerate(ABN_GROUPS)}).fillna(-1).astype(int)
    meta["dur_used"] = np.minimum(meta.n_read / meta.fps, S.MAX_SEC)
    FE = {r: np.load(os.path.join(FE_DIR, k + ".npy")) for r, k in enumerate(meta.key)}
    meta["n_feat"] = [len(FE[r]) for r in range(len(meta))]
    WSTART = {r: S.segments(len(FE[r]), segment=WIN, stride=STRIDE) for r in range(len(meta))}
    print(f"{len(meta)} video | {sum(len(v) for v in WSTART.values())} cua so 16 frame | {meta.n_feat.sum()} frame")
    print("Thoi luong TB (giay):", meta.groupby(["ab", "split"]).dur_used.mean().round(1).to_dict())
    """),
    code("""
    TR_AB = meta.index[(meta.split == "train") & meta.ab & meta.train_use].values
    TR_NO = meta.index[(meta.split == "train") & ~meta.ab & meta.train_use].values
    TE_UCF_AB = meta.index[(meta.split == "test") & meta.ab & (meta.source == "ucf")].values
    TE_UCF_NO = meta.index[(meta.split == "test") & ~meta.ab & (meta.source == "ucf")].values
    TE_RWF = meta.index[(meta.split == "test") & (meta.source == "rwf")].values
    TE_RLVS = meta.index[(meta.split == "test") & (meta.source == "rlvs")].values
    print("== So video")
    print(pd.Series({"train bat thuong (UCF)": len(TR_AB), "train binh thuong (RWF NonFight + RLVS NonViolence)": len(TR_NO),
                     "test bat thuong UCF": len(TE_UCF_AB), "test Normal UCF (ngoai mien)": len(TE_UCF_NO),
                     "test RWF (Fight + NonFight)": len(TE_RWF), "test RLVS (Violence + NonViolence)": len(TE_RLVS)}))
    print("Ky vong: 810 | 1600 | 140 | 150 | - | -   (QUICK chi co mot phan)")
    print("Nhom tren tap train bat thuong:", meta.loc[TR_AB].group.value_counts().to_dict())
    """),
    md("## 4. Chuẩn hoá và cửa sổ"),
    code("""
    samp = np.concatenate([FE[r] for r in np.r_[TR_AB, TR_NO]])
    samp = samp[rng.choice(len(samp), min(len(samp), 50000), replace=False)].astype(np.float32)
    MU, SD = samp.mean(0), np.maximum(samp.std(0), 1e-3)

    def nrm(x):
        return (np.asarray(x, np.float32) - MU) / SD

    def win_x(r, s):
        \"\"\"Cua so 16 frame bat dau tai s. Video < 16 frame: lap frame theo luoi deu (khong dem 0).\"\"\"
        f = FE[r]
        if len(f) >= WIN:
            return f[s:s + WIN]
        return f[np.linspace(0, len(f) - 1, WIN).round().astype(int)]

    def short_of(r):
        return len(FE[r]) < WIN

    def bag_x(r):
        \"\"\"BAGW cua so trai deu theo thoi gian cua 1 video (lap chi so neu video co it cua so).\"\"\"
        st = WSTART[r]
        pick = np.linspace(0, len(st) - 1, BAGW).round().astype(int)
        return np.stack([win_x(r, st[p]) for p in pick])

    def wts(idx):
        w = np.maximum(meta.train_w.values[idx].astype(float), 1e-9)
        return w / w.sum()                                      # can bang theo phut (train_w)
    """),
    code("""
    from sklearn.metrics import roc_auc_score as _auc
    print("== Ti le cua so LAP FRAME (short_clip) va do dai TRUOC khi huan luyen")
    tab = meta.assign(short=[short_of(r) for r in range(len(meta))]).groupby(["ab", "split", "source"]).agg(
        so_video=("key", "size"), ti_le_cua_so_lap=("short", "mean"), do_dai_TB_s=("dur_used", "mean")).round(2)
    display(tab)
    if len(TE_UCF_AB) and len(TE_UCF_NO):
        y_len = np.r_[np.ones(len(TE_UCF_AB)), np.zeros(len(TE_UCF_NO))]
        d_len = meta.dur_used.values[np.r_[TE_UCF_AB, TE_UCF_NO]]
        print(f"AUC CHI DUNG DO DAI tren UCF test (bat thuong vs Normal ngoai mien): {_auc(y_len, d_len):.3f}")
        print(f"Normal UCF test: do dai TB {meta.dur_used.values[TE_UCF_NO].mean():.1f} s | bat thuong test: {meta.dur_used.values[TE_UCF_AB].mean():.1f} s")
    """),
    md("""
    ## 5. Mô hình và huấn luyện (MIL kiểu Sultani trên cửa sổ)
    Mỗi cửa sổ: encoder theo từng frame → trung bình 16 frame → điểm `s` và logit nhóm.
    **Ranking loss:** `max(0, 1 − max s_bất_thường + max s_bình_thường)` trên 8 cửa sổ mỗi video, + làm mượt (`s` cạnh nhau)
    + thưa (`s` bất thường nhỏ). **Head nhóm:** cross-entropy, chỉ trên video bất thường.
    """),
    code("""
    class WinHead(nn.Module):
        def __init__(self, d=960, h=128, p=0.3, n_groups=len(ABN_GROUPS)):
            super().__init__()
            self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p))
            self.score = nn.Linear(h, 1)
            self.group = nn.Linear(h, n_groups)

        def win(self, x):                                       # (B, 16, 960) -> diem (B,), logit nhom (B, 5)
            h = self.enc(x).mean(1)
            return self.score(h).squeeze(-1), self.group(h)

        def bag(self, x):                                       # (B, K, 16, 960) -> diem (B, K), logit nhom (B, 5)
            B, K = x.shape[:2]
            s, g = self.win(x.reshape(B * K, x.shape[2], x.shape[3]))
            return s.view(B, K), g.view(B, K, -1).mean(1)

    model = WinHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_ep, n_st = (QUICK_EPOCHS, QUICK_STEPS) if QUICK else (EPOCHS, STEPS)
    if LOAD_PT:
        ck = torch.load(LOAD_PT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        MU, SD = np.asarray(ck["mu"]), np.asarray(ck["sd"])
        n_ep = 0
        print("Dung scene_head.pt da huan luyen (khong huan luyen lai):", LOAD_PT)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep * n_st)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(n_ep):
        model.train()
        tot = []
        for _ in range(n_st):
            a = rng.choice(TR_AB, BAG, p=wts(TR_AB))
            n = rng.choice(TR_NO, BAG, p=wts(TR_NO))
            xa = torch.from_numpy(nrm(np.stack([bag_x(r) for r in a]))).to(DEVICE)
            xn = torch.from_numpy(nrm(np.stack([bag_x(r) for r in n]))).to(DEVICE)
            sa, ga = model.bag(xa)
            sn, _ = model.bag(xn)
            hinge = torch.relu(1 - sa.max(1).values + sn.max(1).values).mean()
            smooth = ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean()
            sparse = sa.abs().sum(1).mean()
            gi = torch.tensor(meta.gi.values[a], device=DEVICE)
            keep = gi >= 0
            cls_loss = ce(ga[keep], gi[keep]) if keep.any() else torch.zeros((), device=DEVICE)
            loss = hinge + 8e-5 * smooth + 8e-5 * sparse + cls_loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot.append([loss.item(), hinge.item(), cls_loss.item()])
        m = np.mean(tot, 0)
        print(f"epoch {ep + 1}/{n_ep} | loss {m[0]:.4f} | ranking {m[1]:.4f} | nhom {m[2]:.4f} | {time.time() - t0:.0f} s")
    """),
    md("## 6. Đánh giá: hai cách tính điểm video – max cửa sổ và trung bình top-3 cửa sổ"),
    code("""
    @torch.no_grad()
    def video_scores(r):
        model.eval()
        xs = torch.from_numpy(nrm(np.stack([win_x(r, s) for s in WSTART[r]]))).to(DEVICE)
        s, g = model.win(xs)
        return s.cpu().numpy(), g.cpu().numpy()

    VS = {int(r): video_scores(int(r)) for r in np.unique(np.r_[TR_NO, TE_UCF_AB, TE_UCF_NO, TE_RWF, TE_RLVS])}
    vgroup = {r: VS[r][1].mean(0) for r in VS}
    METHODS = {"max": {r: VS[r][0].max() for r in VS},
               "top3": {r: float(np.sort(VS[r][0])[-3:].mean()) for r in VS}}   # ít hơn 3 cửa sổ -> trung bình tất cả

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")

    # NGUONG GOI Y: chi in ra, KHONG ghi vao config (nguon: 95% phan vi video binh thuong TRAIN)
    SUGGEST = {m: float(np.percentile([sc[r] for r in TR_NO], 95)) for m, sc in METHODS.items()}
    print("NGUONG GOI Y (95% phan vi video binh thuong TRAIN = RWF NonFight + RLVS NonViolence; KHONG dat vao config):")
    for m, t in SUGGEST.items():
        print(f"  cach {m:5s}: {t:.3f}")

    TAPS = [("UCF test (bat thuong vs Normal ngoai mien)", TE_UCF_AB, TE_UCF_NO),
            ("RWF test (Fight vs NonFight)", TE_RWF[meta.ab.values[TE_RWF]], TE_RWF[~meta.ab.values[TE_RWF]]),
            ("RLVS test (Violence vs NonViolence)", TE_RLVS[meta.ab.values[TE_RLVS]], TE_RLVS[~meta.ab.values[TE_RLVS]])]
    rows = []
    for m, sc_map in METHODS.items():
        for name, ab_idx, no_idx in TAPS:
            if len(ab_idx) + len(no_idx) == 0:
                continue
            y = np.r_[np.ones(len(ab_idx)), np.zeros(len(no_idx))].astype(int)
            sc = np.array([sc_map[r] for r in np.r_[ab_idx, no_idx]])
            pred = (sc > SUGGEST[m]).astype(int)
            rows.append(dict(cach=m, tap=name, n_bat_thuong=len(ab_idx), n_binh_thuong=len(no_idx), AUC=auc(y, sc),
                             F1_nguong_goi_y=f1_score(y, pred, zero_division=0),
                             FPR_binh_thuong=float(pred[y == 0].mean()) if (y == 0).any() else float("nan")))
    res = pd.DataFrame(rows)
    print("== AUC / F1 / FPR theo cach tinh diem va tung tap test")
    display(res.round(3))
    for m in METHODS:
        if len(TE_UCF_NO):
            fp = float(np.mean([METHODS[m][r] > SUGGEST[m] for r in TE_UCF_NO]))
            print(f"[{m}] TY LE BAO NHAM tren {len(TE_UCF_NO)} Normal UCF (nguong goi y): {fp:.1%}")

    # AUC va FPR theo nhom do dai (UCF test; FPR tinh tren Normal UCF cung nhom do dai)
    BINS = [0, 10, 60, 300, S.MAX_SEC]
    ucf_idx = np.r_[TE_UCF_AB, TE_UCF_NO]
    ucf_y = np.r_[np.ones(len(TE_UCF_AB)), np.zeros(len(TE_UCF_NO))].astype(int)
    ucf_len = meta.dur_used.values[ucf_idx]
    len_rows = []
    for m, sc_map in METHODS.items():
        sc = np.array([sc_map[r] for r in ucf_idx])
        for b in range(len(BINS) - 1):
            in_b = (ucf_len > BINS[b]) & (ucf_len <= BINS[b + 1]) if b else (ucf_len >= 0) & (ucf_len <= BINS[1])
            n_ab, n_no = int((in_b & (ucf_y == 1)).sum()), int((in_b & (ucf_y == 0)).sum())
            len_rows.append(dict(cach=m, do_dai=f"{BINS[b]}-{BINS[b + 1]} s", n_bat_thuong=n_ab, n_binh_thuong=n_no,
                                 AUC_trong_nhom=auc(ucf_y[in_b], sc[in_b]) if (n_ab and n_no) else float("nan"),
                                 FPR_binh_thuong=float((sc[in_b & (ucf_y == 0)] > SUGGEST[m]).mean()) if n_no else float("nan"),
                                 diem_TB_binh_thuong=float(sc[in_b & (ucf_y == 0)].mean()) if n_no else float("nan")))
    lres = pd.DataFrame(len_rows)
    print("== Theo nhom do dai (UCF test): AUC trong nhom, FPR tren Normal UCF, diem TB binh thuong")
    display(lres.round(3))

    # AUC muc cua so (~frame) tren UCF test theo chu thich thoi gian
    spans = {}
    if ANN_F:
        ann = pd.read_csv(ANN_F)
        for _, r in ann.iterrows():
            spans[str(r.fname).lower()] = [(int(r[a]), int(r[b])) for a, b in (("s1", "e1"), ("s2", "e2"))
                                           if r[a] >= 0 and r[b] >= 0]
    fr_y, fr_s = [], []
    for r in np.r_[TE_UCF_AB, TE_UCF_NO]:
        fps_a = float(meta.fps.values[r])
        sp_sec = [(s / fps_a, e / fps_a) for s, e in spans.get(str(meta.fname.values[r]).lower(), [])]
        n = len(FE[r])
        for j, st in enumerate(WSTART[r]):
            t0_, t1_ = st / 2.0, (st + min(WIN, n)) / 2.0          # frame mau 2 FPS -> giay
            fr_y.append(int(any(t0_ < e and t1_ > s for s, e in sp_sec)))
            fr_s.append(VS[r][0][j])
    fr_y, fr_s = np.array(fr_y), np.array(fr_s)
    print(f"AUC MUC CUA SO (~frame) tren UCF test: {auc(fr_y, fr_s):.3f} | cua so duong: {int(fr_y.sum())} / {len(fr_y)}")

    gi_t = meta.gi.values[TE_UCF_AB]
    ok = gi_t >= 0
    pred_g = np.array([vgroup[r].argmax() for r in TE_UCF_AB])
    acc = float((pred_g[ok] == gi_t[ok]).mean()) if ok.any() else float("nan")
    print(f"DO CHINH XAC PHAN NHOM (video bat thuong UCF test, {int(ok.sum())} video): {acc:.1%}")
    print(pd.crosstab(pd.Series(np.array(ABN_GROUPS)[gi_t[ok]], name="that"),
                      pd.Series(np.array(ABN_GROUPS)[pred_g[ok]], name="du doan")))

    print("Ti le cua so NGAN (short_clip) theo nhom:", meta.assign(sh=[short_of(r) for r in range(len(meta))])
          .groupby(["ab", "split"]).sh.mean().round(2).to_dict())
    lens = []
    for name, idx, lab in [("UCF test", ucf_idx, ucf_y),
                           ("RWF test", TE_RWF, meta.ab.values[TE_RWF].astype(int)),
                           ("RLVS test", TE_RLVS, meta.ab.values[TE_RLVS].astype(int))]:
        if len(idx) == 0:
            continue
        sc = np.array([METHODS["max"][r] for r in idx])
        print(f"{name}: AUC chi dung DO DAI = {auc(lab, meta.dur_used.values[idx]):.3f}  (AUC mo hinh cach max = {auc(lab, sc):.3f})")
    """),
    md("## 7. Xuất `scene_head.onnx` (input 16×960), `scene_backbone.onnx`, `scene_config.json`"),
    code(BACKBONE_SRC),
    code("""
    class HeadExport(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):                                   # (B, 16, 960) -> diem (B,), logit nhom (B, 5)
            return self.m.win(x)

    model.eval().cpu()
    torch.save({"model": model.state_dict(), "mu": MU, "sd": SD}, os.path.join(WORK, "scene_head.pt"))
    head_onnx = os.path.join(WORK, "scene_head.onnx")
    torch.onnx.export(HeadExport(model), torch.zeros(2, WIN, 960), head_onnx, input_names=["x"],
                      output_names=["window_score", "group_logits"], opset_version=17,
                      dynamic_axes={"x": {0: "batch"}, "window_score": {0: "batch"}, "group_logits": {0: "batch"}},
                      dynamo=False)
    model.eval()                                            # torch.onnx.export dua lai che do train -> bat dropout
    import onnxruntime as ort
    sess = ort.InferenceSession(head_onnx, providers=["CPUExecutionProvider"])
    xx = nrm(np.stack([win_x(int(r), 0) for r in np.r_[TE_UCF_AB, TR_NO][:4]]))
    ref_s, ref_g = model.win(torch.from_numpy(xx))
    out_s, out_g = sess.run(None, {"x": xx})
    print(f"ONNX head vs PyTorch: lech {np.abs(out_s - ref_s.detach().numpy()).max():.2e} / {np.abs(out_g - ref_g.detach().numpy()).max():.2e}")

    if EXPORT_BACKBONE:
        try:
            bb = SceneBackbone(torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).eval()
        except Exception as e:
            print("!! Khong tai duoc trong so MobileNetV3 (can Internet):", e)
            bb = None
        if bb is not None:
            torch.onnx.export(bb, torch.zeros(2, S.SIZE, S.SIZE, 3, dtype=torch.uint8),
                              os.path.join(WORK, "scene_backbone.onnx"), input_names=["frame_rgb_uint8"],
                              output_names=["feat960"], opset_version=17,
                              dynamic_axes={"frame_rgb_uint8": {0: "batch"}, "feat960": {0: "batch"}}, dynamo=False)
            print("scene_backbone.onnx da xuat (uint8 RGB 224x224 -> 960)")

    cfg_out = {"fps": S.FPS, "size": S.SIZE, "feat_dim": S.FEAT_DIM, "window": WIN, "stride": STRIDE,
               "window_note": "realtime: 16 vector backbone lien tiep (2 FPS = 8 s) -> chuan hoa (mean/std) -> scene_head",
               "groups_abnormal": ABN_GROUPS, "normal_group": NORMAL, "threshold_note": "khong dat nguong cuoi; nguong goi y in trong log danh gia",
               "mean": MU.tolist(), "std": SD.tolist(), "trained_on": "UCF train + RWF NonFight + RLVS NonViolence",
               "quick": QUICK, "results": res.to_dict("records")}
    with open(os.path.join(WORK, "scene_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, ensure_ascii=False, indent=1, default=float)
    for fn in sorted(os.listdir(WORK)):
        if fn.startswith("scene_"):
            print(f"{fn:24s} {os.path.getsize(os.path.join(WORK, fn)) / 1e6:8.2f} MB")
    """),
]

if __name__ == "__main__":
    build(CELLS_05, OUT_05)
