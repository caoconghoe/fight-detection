"""Unit test shared/scene.py (luồng 2) + mục `scene` trong config.yaml.
Chạy (từ thư mục httm):  python realtime/tests/test_scene.py"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import scene as S  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["scene"]
IDX = S.class_index(CFG["groups"], CFG["eval_only"])
K = "/kaggle/input/datasets/"


def test_config_groups_valid():
    seen = {}
    for g, classes in CFG["groups"].items():
        for c in classes:
            assert S.norm(c) not in seen, f"lop {c} nam o 2 nhom: {seen[S.norm(c)]}, {g}"
            seen[S.norm(c)] = g
    assert CFG["normal_group"] in CFG["groups"] and "Normal" in CFG["groups"][CFG["normal_group"]]
    for c, g in CFG["eval_only"].items():
        assert g in CFG["groups"] and S.norm(c) not in seen


def test_detect_class():
    cases = {
        "mission-ai/crimeucfdataset/Part-1/Abuse/Abuse001_x264.mp4": "Abuse",
        "mission-ai/crimeucfdataset/Part-2/Burglary/Burglary010_x264.mp4": "Burglary",
        "mission-ai/crimeucfdataset/Normal-Videos-Part-1/Normal_Videos001_x264.mp4": "Normal",
        "u/UCF_Crimes/Videos/Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4": "Normal",
        "u/Videos/Road_Accidents/RoadAccidents001_x264.mp4": "RoadAccidents",          # thư mục có dấu "_"
        "u/flat/Shoplifting010_x264.mp4": "Shoplifting",                                 # chỉ có tiền tố tên file
        "alirakhmaev/ucf-crime-full/UCF_Crime/RoadAccidents/RoadAccidents150_x264.mp4": "RoadAccidents",
        "alirakhmaev/ucf-crime-full/UCF_Crime/Normal_Videos_for_Event_Recognition/Normal_Videos_015_x264.mp4": "Normal",
        "r/RWF-2000/train/Fight/a_0.avi": "Fight",
        "r/RWF-2000/val/NonFight/b_0.avi": "NonFight",
        "l/real-life-violence-situations-dataset/Real Life Violence Dataset/NonViolence/NV_1.mp4": "NonViolence",
        "u/misc/clip.mp4": None,
    }
    for p, want in cases.items():
        assert S.detect_class(K + p, IDX) == want, (p, S.detect_class(K + p, IDX))
    assert S.detect_class("D:\\data\\UCF\\Arson\\Arson011_x264.mp4", IDX) == "Arson"      # đường dẫn Windows


def test_group_and_train_use():
    assert IDX[S.norm("Fighting")][1:] == ("BẠO LỰC", True)
    assert IDX[S.norm("Fight")][1:] == ("BẠO LỰC", False)                 # RWF Fight: chỉ đánh giá
    assert IDX[S.norm("NonFight")][1:] == ("BÌNH THƯỜNG", True)
    assert S.source_of("Fight") == "rwf" and S.source_of("NonViolence") == "rlvs" and S.source_of("Arson") == "ucf"


def test_skip_and_dup():
    assert S.skip_path(K + "mission-ai/crimeucfdataset/anomaly/Part-1/Abuse/Abuse001_x264.mp4")
    assert not S.skip_path(K + "mission-ai/crimeucfdataset/Part-1/Abuse/Abuse001_x264.mp4")
    assert S.dup_rank(K + "l/real life violence situations/Real Life Violence Dataset/Violence/V_1.mp4") == 1
    assert S.dup_rank(K + "l/Real Life Violence Dataset/Violence/V_1.mp4") == 0
    assert S.video_key("ucf", "Abuse", "/a/b/Abuse 001(1).mp4") == "ucf__Abuse__Abuse_001_1_"


def test_segments():
    assert S.segments(10) == [0]
    assert S.segments(16) == [0]
    assert S.segments(40) == [0, 16, 24]
    assert S.segments(40, stride=8) == [0, 8, 16, 24]


def test_annotation_and_frame_labels():
    ann = S.parse_temporal_annotation("Arson011_x264.mp4  Arson  150  420  600  700\n"
                                      "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n\nhong\n")
    assert ann == {"Arson011_x264.mp4": ("Arson", [(150, 420), (600, 700)]),
                   "Normal_Videos_003_x264.mp4": ("Normal", [])}
    t = np.arange(0, 25, 0.5)                                            # 2 FPS
    y = S.frame_labels(t, 30.0, ann["Arson011_x264.mp4"][1])             # 5–14 s và 20–23.3 s
    assert y[t == 4.5][0] == 0 and y[t == 5.0][0] == 1 and y[t == 14.0][0] == 1 and y[t == 14.5][0] == 0
    assert y[t == 21.0][0] == 1 and y.sum() == 19 + 7
    assert S.parse_test_list("Abuse/Abuse028_x264.mp4\r\nArson\\Arson011_x264.mp4\n") == {
        "Abuse028_x264.mp4", "Arson011_x264.mp4"}


def test_prep():
    f = np.zeros((240, 320, 3), np.uint8)
    f[..., 0] = 255                                                       # BGR: xanh dương
    out = S.prep(f)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert out[..., 2].min() == 255 and out[..., 0].max() == 0            # đã đổi sang RGB


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK   {name}")
        except AssertionError as e:
            failed += 1
            print(f"SAI  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test dat")
    sys.exit(1 if failed else 0)
