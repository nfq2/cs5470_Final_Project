"""The rule: did pushing the wire bring the tip closer to the target along the vessels?

python rule.py geo/tips_clicked.json     # on clicked tips (ground truth, ~30 cases)
python rule.py geo/tips_auto.json --candidate A   # on automatic tips, all cases

Per case: d1, d6 = along-vessel distance (px) from the tip in f1 / f6 to the target;
pushed = mm the wire advanced over the window. Expected if on the right path: d6 < d1 by an
amount proportional to pushed. Features: progress = d1 - d6, ratio = progress / pushed,
d6. Each is scored by AUROC for retract on its own; no fitting, so there is nothing to
overfit on 30 cases.
"""
import argparse
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from geo import DATA_DIR, GEO_DIR
from tip import load_geo

SNAP_PX = 40


def along_vessel_distance(dist, mask, xy):
    """Along-vessel distance at the tip (finite everywhere), and how far the tip is from
    the nearest detected vessel; nan if that is beyond SNAP_PX (tip finder probably missed)."""
    if xy is None:
        return np.nan, np.nan
    x, y = (int(round(v)) for v in xy)
    off_vessel = float(ndi.distance_transform_edt(~mask)[y, x])
    if off_vessel > SNAP_PX:
        return np.nan, off_vessel
    return float(dist[y, x]), off_vessel


def auroc_retract(score, label):
    """AUROC for retract (label 0) with higher score = more retract-like; nan-safe."""
    ok = np.isfinite(score)
    s, l = score[ok], label[ok]
    pos, neg = s[l == 0], s[l == 1]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, int(ok.sum())
    auc = ((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return float(auc), int(ok.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tips", help="JSON list of {case_id, f1, f6} with tips as [x, y]")
    parser.add_argument("--candidate", default=None, help="A or B for automatic tips")
    args = parser.parse_args()
    with open(DATA_DIR / "cases.jsonl") as f:
        cases = {c["case_id"]: c for c in (json.loads(l) for l in f if l.strip())}
    with open(GEO_DIR.parent / args.tips if not args.tips.startswith("/") else args.tips) as f:
        tips = json.load(f)

    rows = []
    for t in tips:
        case = cases[t["case_id"]]
        dist, mask = load_geo(case["case_id"])
        pick = (lambda v: v.get(args.candidate) if isinstance(v, dict) else v)
        d1, snap1 = along_vessel_distance(dist, mask, pick(t["f1"]))
        d6, snap6 = along_vessel_distance(dist, mask, pick(t["f6"]))
        pushed = -case["depth_relative_mm"][0]
        rows.append({"label": case["label"], "d1": d1, "d6": d6, "pushed": pushed,
                     "snap1": snap1, "snap6": snap6})
    label = np.array([r["label"] for r in rows])
    d1 = np.array([r["d1"] for r in rows]); d6 = np.array([r["d6"] for r in rows])
    pushed = np.array([r["pushed"] for r in rows])
    progress = d1 - d6
    print(f"cases: {len(rows)}  advance: {(label == 1).sum()}  retract: {(label == 0).sum()}")
    print(f"tips snapped to a vessel: f1 {np.isfinite(d1).sum()}/{len(rows)}, f6 {np.isfinite(d6).sum()}/{len(rows)}")
    print(f"median snap px: f1 {np.nanmedian([r['snap1'] for r in rows]):.0f}, f6 {np.nanmedian([r['snap6'] for r in rows]):.0f}")
    for name, score in [("-progress (less progress -> retract)", -progress),
                        ("-progress/pushed", -progress / np.maximum(pushed, 1)),
                        ("d6 (farther from target -> retract)", d6),
                        ("d6 - d1 > 0 (moved away)", (d6 - d1 > 0).astype(float))]:
        auc, n = auroc_retract(score, label)
        print(f"  AUROC {auc:.3f} (n={n})  {name}")
    for lab, name in [(1, "advance"), (0, "retract")]:
        m = (label == lab) & np.isfinite(progress)
        if m.any():
            print(f"  {name}: median progress px {np.median(progress[m]):.0f}, "
                  f"median pushed mm {np.median(pushed[m]):.1f}, moved away: {np.mean(progress[m] < 0):.2f}")
