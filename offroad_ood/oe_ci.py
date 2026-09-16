#!/usr/bin/env python
"""Frame-bootstrap 95% CI for the outlier-exposure (MSP OE-trained) main-table row,
from cached OE score maps (no retraining). Same histogram-bootstrap as compute_ci.py.
Self-check: point AP/FPR95 must match the OE json / main table (goose 0.341/0.520).
Writes results/oe_ci.json only."""
import json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
from compute_ci import metrics_from_hist, NBINS, NBOOT
R = f"{ROOT}/results"
DS = ["goose", "rellis", "wildscenes", "rugd"]


def oe_npz(ds, split):
    return f"seghead_scores_oe_l0.5_{split}.npz" if ds == "goose" else f"seghead_scores_{ds}_oe_l0.5_{split}.npz"


def run(ds):
    _, ol = get_dataset(ds); o = ol()
    tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items():
        tl[k] = v
    z = np.load(f"{R}/{oe_npz(ds, 'test')}"); S = z["MSP"].astype(np.float32); gt = tl[z["keys"]]
    F = S.shape[0]
    samp = S[gt != 255][::10]
    edges = np.unique(np.quantile(samp.astype(np.float64), np.linspace(0, 1, NBINS + 1)))
    edges[0] -= 1e-6; edges[-1] += 1e-6; nb = len(edges) - 1
    idh = np.zeros((F, nb)); oodh = np.zeros((F, nb))
    for i in range(F):
        s = S[i]; g = gt[i]
        idh[i] = np.histogram(s[g == 0], bins=edges)[0]
        oodh[i] = np.histogram(s[g == 1], bins=edges)[0]
    pe = metrics_from_hist(idh.sum(0), oodh.sum(0), edges)
    rng = np.random.default_rng(0); boot = {"AP": [], "FPR95": []}
    for _ in range(NBOOT):
        idx = rng.integers(0, F, F)
        r = metrics_from_hist(idh[idx].sum(0), oodh[idx].sum(0), edges)
        boot["AP"].append(r["AP"]); boot["FPR95"].append(r["FPR95"])
    hw = lambda v: float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2)
    return pe, {"AP": hw(boot["AP"]), "FPR95": hw(boot["FPR95"])}


if __name__ == "__main__":
    out = {}
    print(f"{'dataset':11s} {'AP[±hw]':>16s} {'FPR95[±hw]':>16s}")
    for ds in DS:
        pe, hw = run(ds)
        out[ds] = {"point": pe, "ci_halfwidth": hw}
        print(f"{ds:11s} {pe['AP']:.3f}[{hw['AP']:.3f}]".ljust(28) + f" {pe['FPR95']:.3f}[{hw['FPR95']:.3f}]")
    json.dump(out, open(f"{R}/oe_ci.json", "w"), indent=2)
    print("[saved] oe_ci.json")
