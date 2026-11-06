"""Unit test phần nhãn + precision/recall của tools/evaluate.py (không cần video).
Chạy (từ thư mục httm):  python realtime/tests/test_evaluate.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate as E  # noqa: E402


def row(name, events):
    prefix, expected = E.label_of(Path(name))
    ev = [(ty, t, [1]) for ty, t in events]
    return dict(file=name, prefix=prefix, expected=expected, alerted=E.alerted_types(ev), events=ev)


def test_labels_from_prefix():
    assert E.label_of(Path("fight_abc_0.avi")) == ("fight", "ĐÁNH NHAU")
    assert E.label_of(Path("NonFight_x.mp4")) == ("nonfight", None)
    assert E.label_of(Path("handsup_1.mp4"))[1] == "GIƠ TAY ĐẦU HÀNG"
    assert E.label_of(Path("abc.mp4"))[1] == "?"


def test_fall_still_counts_as_fall_and_still():
    assert E.alerted_types([("TÉ NGÃ RỒI NẰM BẤT ĐỘNG", 3.0, [1])]) >= {"TÉ NGÃ", "NẰM BẤT ĐỘNG"}


def test_metrics():
    rows = [row("fight_1.avi", [("ĐÁNH NHAU", 2.0)]), row("fight_2.avi", []),
            row("nonfight_1.avi", [("ĐÁNH NHAU", 1.0)]), row("nonfight_2.avi", []),
            row("fall_1.mp4", [("TÉ NGÃ RỒI NẰM BẤT ĐỘNG", 4.0)]), row("la_1.mp4", [("ĐÁNH NHAU", 1.0)])]
    m = {x["loai"]: x for x in E.metrics(rows)}
    f = m["ĐÁNH NHAU"]
    assert (f["TP"], f["FN"], f["FP"]) == (1, 1, 1) and f["precision"] == 0.5 and f["recall"] == 0.5
    assert f["do_tre_TB_s"] == 2.0 and f["clip_bao_nham"] == "nonfight_1.avi"     # clip nhãn '?' không tính
    assert m["TÉ NGÃ"]["recall"] == 1.0 and m["TÉ NGÃ"]["do_tre_TB_s"] == 4.0
    assert m["NẰM BẤT ĐỘNG"]["FP"] == 1 and m["NẰM BẤT ĐỘNG"]["recall"] is None


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
