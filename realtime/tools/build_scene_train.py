"""Sinh kaggle/05_scene_train.ipynb – huấn luyện LUỒNG 2 (CLAUDE.md mục 12.4).

Đọc output của notebook 04 (Add Input): scene_seg32/ (32 đoạn × 960 cho mỗi video), scene_meta.csv,
ucf_annotation.csv. Train MIL kiểu Sultani (ranking loss) + head phân nhóm (5 nhóm bất thường, chỉ học trên
video bất thường). Xuất scene_head.onnx, scene_backbone.onnx, scene_config.json.
Chạy (từ thư mục httm):  python realtime/tools/build_scene_train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kaggle_notebook import ROOT, code, md, writefile  # noqa: E402
from build_scene_notebooks import BACKBONE_SRC, build, scene_cfg_cell  # noqa: E402

OUT_05 = ROOT / "kaggle" / "05_scene_train.ipynb"

CELLS_05 = [
    md("""
    # 05 – Huấn luyện LUỒNG 2: MIL kiểu Sultani + head phân nhóm (UCF-Crime)

    Sinh tự động bởi `realtime/tools/build_scene_train.py` – **không sửa tay**.

    **Chuẩn bị – Add Input:** output của notebook **04_scene_features** (có `scene_seg32/`, `scene_meta.csv`,
    `ucf_annotation.csv`). Settings → **GPU T4** · **Internet on** (chỉ để cài `onnx` và tải trọng số MobileNetV3 khi xuất
    `scene_backbone.onnx`).

    **Dữ liệu:** train bất thường = UCF (810 video) · train bình thường = RWF NonFight + RLVS NonViolence (~1600) ·
    test = UCF bất thường (140) · Normal UCF (150, **ngoài miền**) · RWF / RLVS val (Fight / Violence chỉ đánh giá).
    Lấy mẫu cân bằng theo `train_w` (theo phút). Không đệm 0: mỗi video đã là 32 đoạn cố định.

    **Chạy:** `QUICK = True` thử → ổn thì `QUICK = False` (Save & Run All).
    """),
    code("""
    import importlib.util, subprocess, sys
    if importlib.util.find_spec("onnx") is None:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxscript"])
        assert r.returncode == 0, "pip install loi -> bat Internet: Settings -> Internet on"
    import onnx
    print("onnx", onnx.__version__)
    """),
    md("## 1. Code dùng chung với realtime"),
    writefile("scene.py"),
    scene_cfg_cell(),
    md("## 2. Cấu hình"),
    code("""
    import os
    QUICK = True              # True: thu nhanh. Ok roi doi False de chay day du.
    QUICK_EPOCHS, QUICK_STEPS = 2, 20
    EPOCHS, STEPS = 20, 200
    BAG = 8                   # moi buoc: BAG video bat thuong + BAG video binh thuong
    LR, WD = 1e-3, 1e-4
    SEED = 42
    EXPORT_BACKBONE = True    # xuat scene_backbone.onnx (can Internet de tai trong so ImageNet)
    INPUT = os.environ.get("HTTM_INPUT", "/kaggle/input")
    WORK = os.environ.get("HTTM_WORK", "/kaggle/working")
    os.makedirs(WORK, exist_ok=True)

    import os, sys, glob, json, time, importlib, shutil
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision
    from sklearn.metrics import roc_auc_score, f1_score
    sys.path.insert(0, os.getcwd())
    import scene as S
    importlib.reload(S)
    CFG = json.load(open("scene_groups.json", encoding="utf-8"))
    NORMAL = CFG["normal_group"]
    ABN_GROUPS = [g for g in CFG["groups"] if g != NORMAL]
    print("5 nhom bat thuong:", ABN_GROUPS)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| device:", DEVICE)
    print(f"QUICK = {QUICK} | MAX_SEC (15 phut khi chay day du) = {S.MAX_SEC} s")
    """),
    md("## 3. Dữ liệu từ notebook 04"),
    code("""
    seg_dirs = sorted(d for d in glob.glob(os.path.join(INPUT, "**", "scene_seg32*"), recursive=True) if os.path.isdir(d))
    assert seg_dirs, "Khong thay scene_seg32/ -> Add Input output cua notebook 04_scene_features"
    full = [d for d in seg_dirs if os.path.basename(d) == "scene_seg32"]
    SEG = (full or seg_dirs)[0]
    print("Doan 32:", SEG, "| cac thu muc tim thay:", seg_dirs)
    META = os.path.join(os.path.dirname(SEG), "scene_meta.csv")
    if not os.path.exists(META):
        META = glob.glob(os.path.join(INPUT, "**", "scene_meta.csv"), recursive=True)[0]
    ANN_F = (glob.glob(os.path.join(INPUT, "**", "ucf_annotation.csv"), recursive=True) or [None])[0]
    meta = pd.read_csv(META)
    meta = meta[meta.key.map(lambda k: os.path.exists(os.path.join(SEG, k + ".npy")))].reset_index(drop=True)
    meta["ab"] = meta.group != NORMAL
    meta["gi"] = meta.group.map({g: i for i, g in enumerate(ABN_GROUPS)}).fillna(-1).astype(int)
    meta["dur_used"] = np.minimum(meta.n_read / meta.fps, S.MAX_SEC)
    print(f"{len(meta)} video co doan 32 | meta: {META}")
    print("Thoi luong trung binh (giay) theo nhom:", meta.groupby(["ab", "split"]).dur_used.mean().round(1).to_dict())
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
    print("Nhan nhom tren tap train bat thuong:", meta.loc[TR_AB].group.value_counts().to_dict())
    """),
    md("## 4. Chuẩn hoá đặc trưng (thống kê trên các đoạn train)"),
    code("""
    XS = np.stack([np.load(os.path.join(SEG, k + ".npy")) for k in meta.key])        # (N, 32, 960) float16
    print("XS", XS.shape, XS.dtype, f"{XS.nbytes / 1e9:.2f} GB")
    tr_idx = np.r_[TR_AB, TR_NO]
    samp = XS[tr_idx].reshape(-1, 960)
    samp = samp[rng.choice(len(samp), min(len(samp), 20000), replace=False)].astype(np.float32)
    MU, SD = samp.mean(0), np.maximum(samp.std(0), 1e-3)

    def norm(x):
        return (x.astype(np.float32) - MU) / SD

    def wts(idx):
        w = np.maximum(meta.train_w.values[idx].astype(float), 1e-9)
        return w / w.sum()                                       # lay mau ti le train_w (can bang theo phut)
    """),
    md("""
    ## 5. Mô hình và huấn luyện (MIL kiểu Sultani)
    Điểm đoạn `s_i` (32 đoạn / video). **Ranking loss:** `max(0, 1 − max_i s_abn + max_j s_nor)` + làm mượt theo thời gian
    + thưa (`s_abn` nhỏ). **Head nhóm:** logit 5 nhóm cấp video, **chỉ học trên video bất thường**.
    """),
    code("""
    class SceneHead(nn.Module):
        def __init__(self, d=960, h=128, p=0.3, n_groups=len(ABN_GROUPS)):
            super().__init__()
            self.enc = nn.Sequential(nn.Dropout(p), nn.Linear(d, h), nn.ReLU(), nn.Dropout(p))
            self.score = nn.Linear(h, 1)
            self.group = nn.Linear(h, n_groups)

        def forward(self, x):                                    # (B, 32, 960) -> diem doan (B, 32), logit nhom (B, 5)
            h = self.enc(x)
            return self.score(h).squeeze(-1), self.group(h.mean(1))

    model = SceneHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_ep, n_st = (QUICK_EPOCHS, QUICK_STEPS) if QUICK else (EPOCHS, STEPS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep * n_st)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(n_ep):
        model.train()
        tot = []
        for _ in range(n_st):
            a = rng.choice(TR_AB, BAG, p=wts(TR_AB))
            n = rng.choice(TR_NO, BAG, p=wts(TR_NO))
            sa, ga = model(torch.from_numpy(norm(XS[a])).to(DEVICE))
            sn, _ = model(torch.from_numpy(norm(XS[n])).to(DEVICE))
            hinge = torch.relu(1 - sa.max(1).values + sn.max(1).values).mean()
            smooth = ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean()
            sparse = sa.abs().sum(1).mean()
            gi = torch.tensor(meta.gi.values[a], device=DEVICE)
            loss = hinge + 8e-5 * smooth + 8e-5 * sparse + ce(ga, gi)     # ga: chi bag bat thuong
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot.append([loss.item(), hinge.item()])
        m = np.mean(tot, 0)
        print(f"epoch {ep + 1}/{n_ep} | loss {m[0]:.4f} | ranking {m[1]:.4f} | {time.time() - t0:.0f} s")
    """),
    md("## 6. Đánh giá"),
    code("""
    @torch.no_grad()
    def scores(idx):
        model.eval()
        S_, G_ = [], []
        for c in range(0, len(idx), 128):
            s, g = model(torch.from_numpy(norm(XS[idx[c:c + 128]])).to(DEVICE))
            S_.append(s.cpu().numpy()); G_.append(g.cpu().numpy())
        return np.concatenate(S_), np.concatenate(G_)

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")

    s_tr_no = scores(TR_NO)[0].max(1)
    THR = float(np.percentile(s_tr_no, 95))          # nguong: 95% diem video binh thuong train
    print(f"Nguong (95% diem binh thuong train) = {THR:.3f}")

    rows = []
    for name, ab_idx, no_idx in [("UCF test (ngoai mien Normal)", TE_UCF_AB, TE_UCF_NO),
                                 ("RWF test (Fight vs NonFight)", TE_RWF[meta.ab.values[TE_RWF]],
                                  TE_RWF[~meta.ab.values[TE_RWF]]),
                                 ("RLVS test (Violence vs NonViolence)", TE_RLVS[meta.ab.values[TE_RLVS]],
                                  TE_RLVS[~meta.ab.values[TE_RLVS]])]:
        idx = np.r_[ab_idx, no_idx]
        if len(idx) == 0:
            continue
        y = np.r_[np.ones(len(ab_idx)), np.zeros(len(no_idx))].astype(int)
        sc = scores(idx)[0].max(1)
        pred = (sc > THR).astype(int)
        rows.append(dict(tap=name, n_bat_thuong=len(ab_idx), n_binh_thuong=len(no_idx), AUC=auc(y, sc),
                         F1=f1_score(y, pred, zero_division=0),
                         bao_nham_tren_binh_thuong=float(pred[y == 0].mean()) if (y == 0).any() else float("nan")))
    res = pd.DataFrame(rows)
    print("== AUC / F1 theo tung tap test (nguong 95% binh thuong train)")
    display(res.round(3))
    if len(TE_UCF_NO):
        fp = float((scores(TE_UCF_NO)[0].max(1) > THR).mean())
        print(f"TY LE BAO NHAM tren {len(TE_UCF_NO)} Normal UCF (ngoai mien): {fp:.1%}")

    # AUC muc doan (~ muc frame) tren UCF test theo chu thich thoi gian
    spans = {}
    if ANN_F:
        ann = pd.read_csv(ANN_F)
        for _, r in ann.iterrows():
            sp = [(int(r[a]), int(r[b])) for a, b in (("s1", "e1"), ("s2", "e2")) if r[a] >= 0 and r[b] >= 0]
            spans[str(r.fname).lower()] = sp
    idx_fr = np.r_[TE_UCF_AB, TE_UCF_NO]
    fr_y, fr_s = [], []
    s_all = scores(idx_fr)[0]
    for j, i in enumerate(idx_fr):
        t = (np.arange(32) + 0.5) / 32 * meta.dur_used.values[i]
        sp = spans.get(str(meta.fname.values[i]).lower(), [])
        fr_y.append(S.frame_labels(t, float(meta.fps.values[i]), sp) if sp else np.zeros(32, np.int8))
        fr_s.append(s_all[j])
    fr_y, fr_s = np.concatenate(fr_y), np.concatenate(fr_s)
    print(f"AUC MUC DOAN (~frame) tren UCF test: {auc(fr_y, fr_s):.3f} | doan duong: {int(fr_y.sum())} / {len(fr_y)}")

    # nhom: dung tren video bat thuong test
    gi_t = meta.gi.values[TE_UCF_AB]
    _, g_logits = scores(TE_UCF_AB)
    ok = gi_t >= 0
    acc = float((g_logits.argmax(1)[ok] == gi_t[ok]).mean()) if ok.any() else float("nan")
    print(f"DO CHINH XAC PHAN NHOM (video bat thuong UCF test, {ok.sum()} video): {acc:.1%}")
    print(pd.crosstab(pd.Series(np.array(ABN_GROUPS)[gi_t[ok]], name="that"),
                      pd.Series(np.array(ABN_GROUPS)[g_logits.argmax(1)[ok]], name="du doan")))

    # kiem tra loi tat do dai: diem TB theo nhom do dai clip, va AUC cua chinh do dai
    lens = []
    for name, idx, lab in [("UCF test", np.r_[TE_UCF_AB, TE_UCF_NO], np.r_[np.ones(len(TE_UCF_AB)), np.zeros(len(TE_UCF_NO))]),
                           ("RWF test", TE_RWF, meta.ab.values[TE_RWF].astype(int)),
                           ("RLVS test", TE_RLVS, meta.ab.values[TE_RLVS].astype(int))]:
        if len(idx) == 0:
            continue
        sc = scores(idx)[0].max(1)
        print(f"{name}: AUC chi dung DO DAI = {auc(lab, meta.dur_used.values[idx]):.3f}  (AUC mo hinh = {auc(lab, sc):.3f})")
        lens.append(pd.DataFrame(dict(tap=name, nhan=lab, do_dai=pd.cut(meta.dur_used.values[idx],
                                                                        [0, 10, 60, 300, S.MAX_SEC]),
                                      diem=sc)))
    if lens:
        print("== Diem trung binh theo do dai clip (kiem tra loi tat)")
        display(pd.concat(lens).groupby(["tap", "nhan", "do_dai"], observed=True).diem.agg(["count", "mean"]).round(3))
    """),
    code(BACKBONE_SRC),
    md("## 7. Xuất `scene_head.onnx`, `scene_backbone.onnx`, `scene_config.json`"),
    code("""
    model.eval().cpu()
    torch.save({"model": model.state_dict(), "mu": MU, "sd": SD}, os.path.join(WORK, "scene_head.pt"))
    head_onnx = os.path.join(WORK, "scene_head.onnx")
    torch.onnx.export(model, torch.zeros(2, 32, 960), head_onnx, input_names=["x"],
                      output_names=["seg_score", "group_logits"], opset_version=17,
                      dynamic_axes={"x": {0: "batch"}, "seg_score": {0: "batch"}, "group_logits": {0: "batch"}},
                      dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(head_onnx, providers=["CPUExecutionProvider"])
    xx = norm(XS[TE_UCF_AB[:4]] if len(TE_UCF_AB) else XS[:4])
    ref_s, ref_g = model(torch.from_numpy(xx))
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

    cfg_out = {"fps": S.FPS, "size": S.SIZE, "feat_dim": S.FEAT_DIM, "n_seg": 32, "segment_note":
               "train tren 32 doan cua video; realtime phai tao 32 doan tu chuoi frame cung cach (xem scene.py)",
               "groups_abnormal": ABN_GROUPS, "normal_group": NORMAL, "threshold": THR,
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
