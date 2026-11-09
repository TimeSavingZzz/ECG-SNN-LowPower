#!/usr/bin/env python3
"""Add a width-matched depth family (prefix truncation) to the model library.

Why: `ab_shallow` (3 blocks) differs from `ab_full` (6 blocks) in width, dilation
AND pool plan at the same time, so the existing 3-vs-6 comparison cannot isolate
depth.  The prefix family keeps every *retained* layer's width/dilation/pool
exactly as in the 6-block model and only removes blocks from the tail:

    L=6  (64,128,128,256,256,384) / (1,2,4,8,16,32) / (1,2,1,2,1,2)   == ab_full
    L=5  (64,128,128,256,256)     / (1,2,4,8,16)    / (1,2,1,2,1)
    L=4  (64,128,128,256)         / (1,2,4,8)       / (1,2,1,2)
    L=3  (64,128,128)             / (1,2,4)         / (1,2,1)

`in_ch` in NeuroCardio follows the plan, and the head is built from `in_ch`,
so truncation needs no other change.

Idempotent: re-running is a no-op.
"""
import pathlib
import sys

NC = pathlib.Path("/mnt/ECG-SNN-LowPower/third_party/neurocardio/neurocardio")
ABL = NC / "ablation.py"
FAC = NC / "factory.py"

CLASS_SRC = '''

# ---- width-matched depth family (prefix truncation, batch 019)
class NeuroCardioPrefix(NeuroCardio):
    """Default 6-block stack truncated to its first ``depth`` blocks.

    Retained layers keep their original width / dilation / pool, so the ONLY
    thing that changes with ``depth`` is how many blocks there are.
    ``depth=6`` is byte-identical to ``ab_full``.
    """

    _PLAN_C = (64, 128, 128, 256, 256, 384)
    _PLAN_D = (1, 2, 4, 8, 16, 32)
    _PLAN_P = (1, 2, 1, 2, 1, 2)

    def __init__(self, num_classes: int = 5, beta: float = 0.9, depth: int = 5):
        L = int(depth)
        assert 1 <= L <= 6, "depth must be in 1..6"
        super().__init__(num_classes=num_classes, beta=beta,
                         channel_plan=self._PLAN_C[:L],
                         dilation_plan=self._PLAN_D[:L],
                         pool_plan=self._PLAN_P[:L])
'''

FAC_ANCHOR = '''    # ---- binary dual-polarity frontend (batch 012) ----'''
FAC_ADD = '''    # ---- width-matched depth family (batch 019) ----
    if name.startswith("ab_pref"):
        from neurocardio.ablation import NeuroCardioPrefix
        return NeuroCardioPrefix(num_classes=num_classes,
                                 depth=int(name[len("ab_pref"):]))

''' + FAC_ANCHOR


def main() -> int:
    ok = True

    txt = ABL.read_text()
    if "class NeuroCardioPrefix" in txt:
        print("[SKIP] ablation.py already has NeuroCardioPrefix")
    else:
        txt = txt.rstrip("\n") + "\n" + CLASS_SRC
        ABL.write_text(txt)
        print("[OK] ablation.py += NeuroCardioPrefix")

    txt = FAC.read_text()
    if "ab_pref" in txt:
        print("[SKIP] factory.py already routes ab_pref*")
    else:
        if txt.count(FAC_ANCHOR) != 1:
            print("[FAIL] factory anchor count = %d" % txt.count(FAC_ANCHOR))
            ok = False
        else:
            FAC.write_text(txt.replace(FAC_ANCHOR, FAC_ADD, 1))
            print("[OK] factory.py += ab_pref*")

    # self-check: build the three configs and report param counts
    sys.path.insert(0, "/mnt/ECG-SNN-LowPower/third_party/neurocardio")
    from neurocardio.factory import make_model
    for L in (3, 4, 5, 6):
        m = make_model("ab_pref%d" % L, num_classes=5)
        n = sum(p.numel() for p in m.parameters())
        shapes = [b.conv.out_channels for b in m.blocks]
        print("  ab_pref%d  params=%d  block_out_channels=%s" % (L, n, shapes))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
