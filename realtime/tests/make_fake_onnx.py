"""Sinh tests/fixtures/fake_pair_tcn.onnx – mô hình GIẢ cùng giao diện pair_tcn.onnx (input x (batch, 32, 180),
output logit (batch,)) để test nhánh ONNX của detectors/fight.py mà không cần mô hình thật.
logit = 100 × max |vận tốc cổ tay| (của A và B, trong cửa sổ) − 5  -> vung tay mạnh: logit cao; đứng yên: thấp.
Cần torch + onnx (chỉ để sinh file; test chỉ cần onnxruntime):  python realtime/tests/make_fake_onnx.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import features as F  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures" / "fake_pair_tcn.onnx"
# cột vận tốc (x, y) của cổ tay trái/phải (khớp 9, 10) của A (102..135) và B (136..169)
WRIST_VEL = [base + 2 * j + c for base in (102, 136) for j in (9, 10) for c in (0, 1)]


class FakePairTCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor(WRIST_VEL))

    def forward(self, x):
        v = torch.index_select(x, 2, self.idx).abs()
        return v.amax(dim=(1, 2)) * 100.0 - 5.0


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    torch.onnx.export(FakePairTCN(), torch.zeros(2, F.WINDOW, F.FEAT_DIM), str(OUT), input_names=["x"],
                      output_names=["logit"], opset_version=17,
                      dynamic_axes={"x": {0: "batch"}, "logit": {0: "batch"}}, dynamo=False)
    print("Da ghi", OUT, OUT.stat().st_size, "byte")
