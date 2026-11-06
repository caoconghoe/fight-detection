"""Sinh kaggle/02_train.ipynb (CLAUDE.md mục 7.2, 7.3).

Ô đầu dùng %%writefile sampling.py / features.py với nội dung đọc TRỰC TIẾP từ realtime/shared/
-> code trích đặc trưng trên Kaggle luôn giống hệt realtime.
Chạy (từ thư mục httm):  python realtime/tools/build_kaggle_notebook.py
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "realtime" / "shared"
OUT = ROOT / "kaggle" / "02_train.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}


def _lines(s):
    s = textwrap.dedent(s).strip("\n")
    lines = s.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def writefile(name):
    return code(f"%%writefile {name}\n" + (SHARED / name).read_text(encoding="utf-8").rstrip("\n"))


CELLS = [
    md("""
    # 02 – Train Pair-TCN + MIL phát hiện ĐÁNH NHAU

    Sinh tự động bởi `realtime/tools/build_kaggle_notebook.py` – **không sửa tay**, sửa ở repo rồi sinh lại.

    **Chuẩn bị:** Add Input → *Notebook Output* của `01_extract_pose` · Settings → Accelerator **GPU T4**.

    **Chạy:** để `QUICK = True` chạy thử (200 clip, 3 epoch, vài phút) → không lỗi thì đổi `QUICK = False`,
    *Save Version → Save & Run All* để train đầy đủ (40 epoch, 3 thí nghiệm).
    Ở QUICK chỉ kiểm tra **chạy không lỗi**: F1/accuracy val dao động mạnh là bình thường (rất ít bước train,
    BatchNorm chưa ổn định) – chỉ đọc kết quả của bản train đầy đủ.

    **Đầu ra (/kaggle/working):** `best_<thí nghiệm>.pt`, `pair_tcn.onnx`, `fight_config.json`,
    `results.csv`, `instance_coverage.csv`, `clip_stats.csv`.
    """),
    md("## 1. Code dùng chung với realtime (`shared/sampling.py`, `shared/features.py`)"),
    writefile("sampling.py"),
    writefile("features.py"),
    md("## 2. Cấu hình"),
    code("""
    QUICK = True            # True: chạy thử nhanh. Chạy ổn rồi đổi False để train đầy đủ.
    QUICK_CLIPS = 200       # số clip khi QUICK (chia đều dataset × split × nhãn)
    EPOCHS = 3 if QUICK else 40
    BATCH_BAGS = 16         # số bag (clip) / batch
    LR, WEIGHT_DECAY = 1e-3, 1e-4
    MAX_INST_TRAIN = 64     # tối đa instance / bag khi train (clip RLVS dài có thể > 200 instance)
    THRESHOLD = 0.6         # ghi vào fight_config.json (ngưỡng realtime, CLAUDE.md 7.4)
    SEED = 42
    EXPERIMENTS = [("RWF", ["RWF"]), ("RLVS", ["RLVS"]), ("GOP", ["RWF", "RLVS"])]   # (tên, dataset dùng để train)
    EXPORT_FROM = "GOP"     # thí nghiệm dùng để xuất pair_tcn.onnx

    import os, sys, glob, json, time, math, random, importlib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    INPUT = os.environ.get("HTTM_INPUT", "/kaggle/input")
    WORK = os.environ.get("HTTM_WORK", "/kaggle/working")
    N_WORKERS = int(os.environ.get("HTTM_WORKERS", os.cpu_count() or 2))
    os.makedirs(WORK, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    import sampling, features as F
    importlib.reload(sampling); importlib.reload(F)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| device:", DEVICE,
          "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "KHONG CO GPU (Settings -> Accelerator)")
    print("QUICK =", QUICK, "| EPOCHS =", EPOCHS, "| CPU workers =", N_WORKERS)
    print("features: FPS", F.FPS, "WINDOW", F.WINDOW, "STRIDE", F.STRIDE, "FEAT_DIM", F.FEAT_DIM)
    """),
    md("## 3. Tìm dữ liệu (đầu ra của `01_extract_pose`)"),
    code("""
    metas = sorted(glob.glob(os.path.join(INPUT, "**", "meta.csv"), recursive=True))
    print("meta.csv tim thay:", metas)
    assert metas, "Khong thay meta.csv trong /kaggle/input -> Add Input -> Notebook Output cua 01_extract_pose"
    META = metas[0]
    if len(metas) > 1:
        print("!! Co nhieu meta.csv, dung file dau tien:", META)
    POSES = os.path.join(os.path.dirname(META), "poses")
    if not os.path.isdir(POSES):
        dirs = [d for d in glob.glob(os.path.join(INPUT, "**", "poses"), recursive=True) if os.path.isdir(d)]
        assert dirs, "Khong thay thu muc poses/"
        POSES = dirs[0]
    print("META  =", META)
    print("POSES =", POSES, "|", len(glob.glob(os.path.join(POSES, "*.npz"))), "file .npz")
    """),
    code("""
    meta = pd.read_csv(META)
    print("Cot:", meta.columns.tolist())
    display(meta.head(3))

    def norm_ds(v):
        s = str(v).lower()
        if "rwf" in s:
            return "RWF"
        if "rlvs" in s or "real" in s or "violence" in s:
            return "RLVS"
        raise ValueError(f"Khong nhan ra dataset: {v!r}")

    def norm_label(v):
        s = str(v).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if s in ("1", "1.0", "fight", "violence", "true"):
            return 1
        if s in ("0", "0.0", "nonfight", "nonviolence", "false"):
            return 0
        raise ValueError(f"Khong nhan ra nhan: {v!r}")

    def norm_split(v):
        s = str(v).strip().lower()
        if s.startswith("train"):
            return "train"
        if s.startswith(("val", "test")):
            return "val"
        raise ValueError(f"Khong nhan ra split: {v!r}")

    meta["ds"] = meta["dataset"].map(norm_ds)
    meta["y"] = meta["label"].map(norm_label).astype(int)
    meta["sp"] = meta["split"].map(norm_split)
    meta["path"] = meta["file"].map(lambda f: os.path.join(POSES, os.path.basename(str(f))))
    ok = meta["path"].map(os.path.exists)
    print(f"Thieu file npz: {(~ok).sum()} / {len(meta)}")
    if (~ok).any():
        display(meta.loc[~ok, ["file", "dataset", "label", "split"]].head())
    meta = meta[ok].reset_index(drop=True)
    print("Gia tri goc -> chuan hoa:")
    display(meta.groupby(["dataset", "label", "split"]).size().rename("so clip").reset_index()
            .assign(ds=lambda d: d.dataset.map(norm_ds), y=lambda d: d.label.map(norm_label),
                    sp=lambda d: d.split.map(norm_split)))

    if QUICK:
        per = max(1, QUICK_CLIPS // meta.groupby(["ds", "sp", "y"]).ngroups)
        take = [i for _, g in meta.groupby(["ds", "sp", "y"])
                for i in g.sample(min(len(g), per), random_state=SEED).index]
        meta = meta.loc[sorted(take)].reset_index(drop=True)
        print(f"QUICK: lay {len(meta)} clip")

    z = np.load(meta.path.iloc[0])
    print("npz mau:", {k: (z[k].shape, z[k].dtype) for k in z.files})
    """),
    md("""
    ## 4. Dựng instance (cặp × cửa sổ)
    Mỗi clip: npz → theo track (pixel) → **lấy mẫu 15 FPS theo thời gian** → nối tracklet → cửa sổ 32 / bước 8 →
    cặp ứng viên. Lưu **chuỗi khung xương thô** của mỗi instance (để tăng cường dữ liệu trước khi tính đặc trưng).
    """),
    code("""
    from multiprocessing import Pool

    def process(path):
        z = np.load(path)
        npz = {k: z[k] for k in ("data", "fps", "n_frames", "width", "height")}
        inst, tracks_k = F.clip_instances(npz)
        K = len(sampling.sample_indices(list(F.clip_times(npz)), F.FPS))   # số frame mẫu của clip
        st = dict(dur=float(npz["n_frames"]) / float(npz["fps"]), K=K,
                  n_win=max(1, (K - F.WINDOW) // F.STRIDE + 1), n_inst=len(inst), n_tracks=len(tracks_k),
                  n_win_pair=len({s for _, _, s, _, _ in inst}))
        W = F.WINDOW
        kp = np.zeros((len(inst), 2, W, 17, 3), np.float32)
        box = np.zeros((len(inst), 2, W, 4), np.float32)
        valid = np.zeros((len(inst), 2, W), bool)
        for i, (_, _, _, A, B) in enumerate(inst):
            for j, s in enumerate((A, B)):
                kp[i, j], box[i, j], valid[i, j] = s["kpts"], s["box"], s["valid"]
        return kp, box, valid, st

    t0 = time.time()
    paths = meta.path.tolist()
    if N_WORKERS > 1:
        with Pool(N_WORKERS) as pool:
            res = pool.map(process, paths, chunksize=8)
    else:
        res = [process(p) for p in paths]
    BUILD_SEC = time.time() - t0

    KP = np.concatenate([r[0] for r in res])
    BOX = np.concatenate([r[1] for r in res])
    VALID = np.concatenate([r[2] for r in res])
    stats = pd.DataFrame([r[3] for r in res])
    del res
    meta = pd.concat([meta, stats], axis=1)
    CLIP_OF = np.repeat(np.arange(len(meta)), meta.n_inst.values)            # instance -> clip
    starts = np.concatenate([[0], np.cumsum(meta.n_inst.values)])
    INST_OF = [np.arange(starts[c], starts[c + 1]) for c in range(len(meta))]  # clip -> các instance
    LABEL = meta.y.values
    meta["has_inst"] = meta.n_inst > 0
    print(f"Dung {len(KP)} instance tu {len(meta)} clip trong {BUILD_SEC:.1f} s "
          f"({BUILD_SEC / len(meta) * 1000:.0f} ms/clip) | bo nho {KP.nbytes / 1e9:.2f} GB")
    """),
    md("## 5. Kiểm tra dữ liệu"),
    code("""
    def feats_of(idx):
        A = {"kpts": KP[idx, 0], "box": BOX[idx, 0], "valid": VALID[idx, 0]}
        B = {"kpts": KP[idx, 1], "box": BOX[idx, 1], "valid": VALID[idx, 1]}
        return F.pair_features(A, B)

    lab = meta.y.map({0: "0 NonFight", 1: "1 Fight"})
    print("== So clip theo dataset / split / nhan")
    display(pd.crosstab([meta.ds, meta.sp], lab, margins=True))
    print("== So clip co >= 1 instance cap")
    display(pd.crosstab([meta.ds, meta.sp], lab, values=meta.has_inst, aggfunc="sum", margins=True))
    print(f"== Tong so instance: {len(KP)} | thoi gian dung instance: {BUILD_SEC:.1f} s")
    display(meta.groupby(["ds", "sp", lab])["n_inst"].agg(["sum", "mean", "median", "max"]).round(1))
    """),
    md("""
    ### Kiểm tra lệch FPS (RLVS: Violence ~29.6 FPS, NonViolence ~24.9 FPS)
    Mọi đặc trưng tính trên trục thời gian đã lấy mẫu về **15 FPS** → *FPS sau lấy mẫu* phải ≈ 15 ở mọi nhóm,
    số cửa sổ/clip chỉ phụ thuộc **thời lượng**, vận tốc tính theo **chiều cao người / giây**.
    Nếu FPS sau lấy mẫu khác nhau giữa 2 nhãn → mô hình có thể học "đường tắt" theo FPS → phải sửa trước khi train.
    """),
    code("""
    meta["fps_sampled"] = meta.K / meta.dur
    print("== FPS goc va FPS sau lay mau (phai ~15 o moi nhom)")
    display(meta.groupby(["ds", lab])[["fps", "fps_sampled", "dur"]].agg(["mean", "min", "max"]).round(2))
    print("== So cua so / clip (truc 15 FPS)")
    display(meta.groupby(["ds", lab])["n_win"].describe().round(1))

    rng = np.random.default_rng(SEED)
    rows = []
    for (ds, y), g in meta.groupby(["ds", "y"]):
        idx = np.concatenate([INST_OF[c] for c in g.index]) if g.n_inst.sum() else np.array([], int)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(len(idx), 3000), replace=False)
        f = feats_of(np.sort(idx))
        v = f[..., F.SL_VEL].reshape(*f.shape[:2], 34, 2)
        conf = f[..., F.SL_CONF] > 0
        speed = np.linalg.norm(v, axis=-1)[conf] * F.FPS                     # chiều cao người / giây
        wr = np.linalg.norm(v[:, :, [9, 10, 26, 27]], axis=-1)[conf[:, :, [9, 10, 26, 27]]] * F.FPS
        rows.append(dict(ds=ds, nhan=y, n_inst=len(idx), van_toc_TB=speed.mean(), van_toc_trung_vi=np.median(speed),
                         van_toc_co_tay_TB=wr.mean(), kc_tam_TB=np.median(f[..., 178][f[..., 178] < F.FAR])))
    print("== Van toc khop trung binh theo nhan (chieu cao nguoi / giay, tinh tren truc 15 FPS)")
    display(pd.DataFrame(rows).round(3))
    """),
    md("### Giới hạn trên của recall: tỉ lệ clip ĐÁNH NHAU có ≥ 1 instance (clip không có cặp nào sẽ luôn bị đoán 0)"),
    code("""
    cov = (meta[meta.y == 1].groupby(["ds", "sp"])
               .agg(clip_fight=("y", "size"), co_instance=("has_inst", "sum"), recall_toi_da=("has_inst", "mean"))
               .reset_index())
    cov = pd.concat([cov, meta[meta.y == 1].groupby("sp").agg(clip_fight=("y", "size"), co_instance=("has_inst", "sum"),
                     recall_toi_da=("has_inst", "mean")).reset_index().assign(ds="GOP")], ignore_index=True)
    display(cov.round(3))
    cov.to_csv(os.path.join(WORK, "instance_coverage.csv"), index=False)
    meta.drop(columns=["path"]).to_csv(os.path.join(WORK, "clip_stats.csv"), index=False)
    """),
    md("""
    ## 6. Tăng cường dữ liệu, mô hình, MIL
    Tăng cường trên **chuỗi khung xương thô** (rồi mới tính đặc trưng bằng `features.pair_features`):
    co giãn thời gian 0.8–1.2 · hoán đổi A/B · lật ngang · nhiễu Gauss σ = 0.01 × chiều cao · xoá 10% khớp ·
    30% xoá thân dưới (khớp 11–16, giả lập webcam thấy nửa người).
    """),
    code("""
    def augment(kp, box, valid, rng):
        kp, box, valid = kp.copy(), box.copy(), valid.copy()
        n, _, W = valid.shape
        # co giãn thời gian quanh tâm cửa sổ (nội suy tuyến tính)
        r = rng.uniform(0.8, 1.2, n)
        c = (W - 1) / 2
        pos = np.clip(c + (np.arange(W)[None] - c) * r[:, None], 0, W - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, W - 1)
        w = (pos - i0).astype(np.float32)
        rows = np.arange(n)[:, None]
        def interp(a):                                   # a (n, 2, W, ...) -> nội suy theo trục W
            t = np.moveaxis(a, 2, 1)
            ww = w.reshape(n, W, *([1] * (t.ndim - 2)))
            return np.moveaxis(t[rows, i0] * (1 - ww) + t[rows, i1] * ww, 1, 2)
        kp, box = interp(kp), interp(box)
        vt = np.moveaxis(valid, 2, 1)
        valid = np.moveaxis(vt[rows, i0] & vt[rows, i1], 1, 2)
        # hoán đổi A/B
        s = rng.random(n) < 0.5
        kp[s], box[s], valid[s] = kp[s][:, ::-1], box[s][:, ::-1], valid[s][:, ::-1]
        # lật ngang: x -> -x, đổi khớp trái/phải
        f = rng.random(n) < 0.5
        kp[f] = kp[f][..., F.FLIP_KPTS, :]
        kp[f, ..., 0] *= -1
        box[f] = np.stack([-box[f][..., 2], box[f][..., 1], -box[f][..., 0], box[f][..., 3]], -1)
        # nhiễu Gauss σ = 0.01 × chiều cao TB của cặp
        h = box[..., 3] - box[..., 1]
        S = np.maximum((h * valid).sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1), 1.0)
        kp[..., :2] += rng.normal(0, 0.01, kp[..., :2].shape).astype(np.float32) * S[:, None, None, None, None]
        # xoá ngẫu nhiên 10% khớp; 30% xoá thân dưới
        kp[..., 2][rng.random(kp.shape[:-1]) < 0.1] = 0
        low = rng.random(n) < 0.3
        kp[low, :, :, 11:17, 2] = 0
        return kp, box, valid

    class BagDS(torch.utils.data.Dataset):
        def __init__(self, clips, train, mean, std):
            self.clips, self.train, self.mean, self.std = list(clips), train, mean, std
        def __len__(self):
            return len(self.clips)
        def __getitem__(self, i):
            c = self.clips[i]
            idx = INST_OF[c]
            if self.train:
                rng = np.random.default_rng()
                if len(idx) > MAX_INST_TRAIN:
                    idx = np.sort(rng.choice(idx, MAX_INST_TRAIN, replace=False))
                kp, box, valid = augment(KP[idx], BOX[idx], VALID[idx], rng)
                A = {"kpts": kp[:, 0], "box": box[:, 0], "valid": valid[:, 0]}
                B = {"kpts": kp[:, 1], "box": box[:, 1], "valid": valid[:, 1]}
                f = F.pair_features(A, B)
            else:
                f = feats_of(idx)
            f = (f - self.mean) / self.std
            return torch.from_numpy(f.astype(np.float32)), float(LABEL[c])

    def collate(batch):
        return (torch.cat([f for f, _ in batch]), torch.tensor([len(f) for f, _ in batch]),
                torch.tensor([y for _, y in batch], dtype=torch.float32))

    class Block(nn.Module):
        def __init__(self, c, d, p=0.2):
            super().__init__()
            self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.bn = nn.BatchNorm1d(c)
            self.drop = nn.Dropout(p)
        def forward(self, x):
            return x + self.drop(torch.relu(self.bn(self.conv(x))))

    class PairTCN(nn.Module):
        def __init__(self, d_in=F.FEAT_DIM, c=128):
            super().__init__()
            self.inp = nn.Conv1d(d_in, c, 1)
            self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8)])
            self.head = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):                            # x (batch, 32, 180) -> logit (batch,)
            h = self.blocks(self.inp(x.transpose(1, 2)))
            return self.head(torch.cat([h.amax(-1), h.mean(-1)], 1)).squeeze(-1)

    def bag_logits(logits, sizes):
        \"\"\"MIL: điểm bag = trung bình top-k logit instance, k = max(1, ceil(n / 8)).\"\"\"
        out = []
        for part in torch.split(logits, sizes.tolist()):
            k = max(1, math.ceil(len(part) / 8))
            out.append(part.topk(k).values.mean())
        return torch.stack(out)

    def norm_stats(clips, max_inst=20000):
        idx = np.concatenate([INST_OF[c] for c in clips])
        idx = np.sort(np.random.default_rng(SEED).choice(idx, min(len(idx), max_inst), replace=False))
        f = feats_of(idx).reshape(-1, F.FEAT_DIM)
        return f.mean(0).astype(np.float32), np.maximum(f.std(0), 1e-3).astype(np.float32)

    @torch.no_grad()
    def predict(model, clips, mean, std):
        \"\"\"Xác suất ĐÁNH NHAU mức clip; clip không có instance -> 0.\"\"\"
        model.eval()
        probs = np.zeros(len(clips), np.float32)
        has = [i for i, c in enumerate(clips) if len(INST_OF[c])]
        ds = BagDS([clips[i] for i in has], False, mean, std)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate, num_workers=0)
        out = []
        for x, sizes, _ in dl:
            out.append(torch.sigmoid(bag_logits(model(x.to(DEVICE)), sizes)).cpu())
        if has:
            probs[has] = torch.cat(out).numpy()
        return probs

    def metrics(y, p, thr=0.5):
        pred = (p >= thr).astype(int)
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
        return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0), auc=auc)

    m = PairTCN()
    print(f"PairTCN: {sum(p.numel() for p in m.parameters()) / 1e3:.0f}k tham so | thu forward:",
          m(torch.zeros(4, F.WINDOW, F.FEAT_DIM)).shape)
    _kp, _box, _valid = augment(KP[:8], BOX[:8], VALID[:8], np.random.default_rng(0))
    print("thu augment:", _kp.shape, _box.shape, _valid.shape)
    """),
    md("## 7. Train + đánh giá: train RWF → test RLVS, train RLVS → test RWF, train gộp"),
    code("""
    def run_experiment(tag, train_ds):
        sel = meta.ds.isin(train_ds)
        tr = meta[sel & (meta.sp == "train")]
        drop1 = int(((tr.y == 1) & ~tr.has_inst).sum())
        drop0 = int(((tr.y == 0) & ~tr.has_inst).sum())
        tr_clips = tr.index[tr.has_inst.values].tolist()
        va_clips = meta.index[sel & (meta.sp == "val")].tolist()
        print(f"\\n===== {tag}: train tren {train_ds} | {len(tr_clips)} bag train "
              f"(bo {drop1} clip Fight + {drop0} clip NonFight khong co instance) | {len(va_clips)} clip val")
        mean, std = norm_stats(tr_clips)
        dl = torch.utils.data.DataLoader(BagDS(tr_clips, True, mean, std), batch_size=BATCH_BAGS, shuffle=True,
                                         collate_fn=collate, num_workers=N_WORKERS if N_WORKERS > 1 else 0,
                                         drop_last=len(tr_clips) > BATCH_BAGS, persistent_workers=N_WORKERS > 1)
        model = PairTCN().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * max(1, len(dl)))
        loss_fn = nn.BCEWithLogitsLoss()
        best, ckpt = -1.0, os.path.join(WORK, f"best_{tag}.pt")
        yv = LABEL[va_clips]
        for ep in range(1, EPOCHS + 1):
            model.train()
            t0, tot, nb = time.time(), 0.0, 0
            for x, sizes, y in dl:
                logit = bag_logits(model(x.to(DEVICE)), sizes)
                loss = loss_fn(logit, y.to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot, nb = tot + loss.item(), nb + 1
            mv = metrics(yv, predict(model, va_clips, mean, std))
            mark = ""
            if mv["f1"] > best:
                best, mark = mv["f1"], "  *luu"
                torch.save({"model": model.state_dict(), "mean": mean, "std": std, "epoch": ep, "tag": tag,
                            "train_ds": train_ds, "val": mv}, ckpt)
            print(f"[{tag}] epoch {ep:2d}/{EPOCHS} loss {tot / max(nb, 1):.4f} | val acc {mv['acc']:.3f} "
                  f"f1 {mv['f1']:.3f} auc {mv['auc']:.3f} | {time.time() - t0:.0f} s{mark}")
        ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        rows = []
        for test in ("RWF", "RLVS", "GOP"):
            ts = meta.index[(meta.sp == "val") & (meta.ds.isin(["RWF", "RLVS"] if test == "GOP" else [test]))].tolist()
            if not ts:
                continue
            p = predict(model, ts, ck["mean"], ck["std"])
            y = LABEL[ts]
            thr = np.linspace(0.05, 0.95, 19)
            f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in thr]
            ub = meta.loc[ts].query("y == 1").has_inst.mean()
            rows.append(dict(train=tag, test=f"{test} val", n_clip=len(ts), **metrics(y, p),
                             f1_tot_nhat=max(f1s), nguong_tot_nhat=thr[int(np.argmax(f1s))],
                             recall_toi_da=ub, best_epoch=ck["epoch"]))
        return model, ck, rows

    RESULTS, MODELS = [], {}
    for tag, train_ds in EXPERIMENTS:
        if not meta.ds.isin(train_ds).any():
            print("Bo qua", tag, "(khong co du lieu)")
            continue
        model, ck, rows = run_experiment(tag, train_ds)
        MODELS[tag] = (model, ck)
        RESULTS += rows
    res = pd.DataFrame(RESULTS)
    res.to_csv(os.path.join(WORK, "results.csv"), index=False)
    print("\\n===== KET QUA (muc clip, nguong 0.5; recall_toi_da = ti le clip Fight co >= 1 instance)")
    display(res.round(3))
    """),
    md("## 8. Xuất `pair_tcn.onnx` + `fight_config.json`"),
    code("""
    model, ck = MODELS[EXPORT_FROM]
    model = model.eval().cpu()
    onnx_path = os.path.join(WORK, "pair_tcn.onnx")
    dummy = torch.zeros(2, F.WINDOW, F.FEAT_DIM)
    kw = dict(input_names=["x"], output_names=["logit"], opset_version=17,
              dynamic_axes={"x": {0: "batch"}, "logit": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kw)
    except TypeError:                                    # torch cũ chưa có tham số dynamo
        torch.onnx.export(model, dummy, onnx_path, **kw)

    cfg = {"fps": F.FPS, "window": F.WINDOW, "stride": F.STRIDE, "feat_dim": F.FEAT_DIM,
           "mean": ck["mean"].tolist(), "std": ck["std"].tolist(), "threshold": THRESHOLD,
           "trained_on": ck["train_ds"], "best_epoch": ck["epoch"], "val": {k: float(v) for k, v in ck["val"].items()},
           "input_note": "x = (pair_features - mean) / std; output logit theo instance, sigmoid -> xac suat"}
    with open(os.path.join(WORK, "fight_config.json"), "w") as f:
        json.dump(cfg, f)

    # kiểm tra ONNX cho cùng kết quả với PyTorch
    idx = INST_OF[int(np.flatnonzero(meta.has_inst)[0])][:8]
    x = ((feats_of(idx) - ck["mean"]) / ck["std"]).astype(np.float32)
    ref = model(torch.from_numpy(x)).detach().numpy()
    try:
        import onnxruntime as ort
        out = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(["logit"], {"x": x})[0]
        print(f"ONNX vs PyTorch: lech toi da {np.abs(out - ref).max():.2e} (batch {len(x)})")
    except ImportError:
        print("(khong co onnxruntime tren Kaggle -> bo qua buoc so sanh; se kiem tra tren may)")
    for fn in sorted(os.listdir(WORK)):
        print(f"{fn:28s} {os.path.getsize(os.path.join(WORK, fn)) / 1e6:8.2f} MB")
    """),
]


def main():
    nb = {"cells": CELLS, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Da ghi {OUT} ({len(CELLS)} o)")


if __name__ == "__main__":
    main()
