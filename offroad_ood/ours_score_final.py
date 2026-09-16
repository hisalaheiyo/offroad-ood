#!/usr/bin/env python
"""
Ours-Score (fixed beta=1 fusion) final numbers for the main table + competitor.
AP, FPR95, and histogram-bootstrap 95% CI (SAME method as compute_ci.py).
Uses m2f_rba MSP-M2F + cdnp (consistent with the existing main-table pipeline;
rugd_comp also available). z-norm on validation-ID pixels. beta=1 fixed (ID-only).

Self-check: beta=0 -> MSP-M2F AP (must match main table MSP-M2F row);
            beta=1 AP -> must match tcc_fused (goose 0.822, rellis 0.588, ws 0.744, rugd 0.948).
Writes results/ours_score_final.json only.
"""
import json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
from compute_ci import metrics_from_hist, NBINS, NBOOT

R = f"{ROOT}/results"
DS_ALL = ["goose", "rellis", "wildscenes", "rugd", "rugd_comp"]


def load_fused(ds, beta):
    pfx = "" if ds == "goose" else f"{ds}_"
    _, ol = get_dataset(ds); o = ol()
    tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    rv = np.load(f"{R}/m2f_rba_scores_{pfx}val.npz"); rt = np.load(f"{R}/m2f_rba_scores_{pfx}test.npz")
    cv = np.load(f"{R}/cdnp_scores_{pfx}val.npz"); ct = np.load(f"{R}/cdnp_scores_{pfx}test.npz")
    assert np.array_equal(rv["keys"], cv["keys"]), f"{ds} val keys misalign (rba vs cdnp)"
    assert np.array_equal(rt["keys"], ct["keys"]), f"{ds} test keys misalign (rba vs cdnp)"
    mv = rv["MSP_M2F"].astype(np.float32).ravel(); cvs = cv["scores"].astype(np.float32).ravel()
    gv = tl[rv["keys"]].ravel(); idv = gv == 0
    muM, sdM = mv[idv].mean(), mv[idv].std() + 1e-6
    muC, sdC = cvs[idv].mean(), cvs[idv].std() + 1e-6
    mt = rt["MSP_M2F"].astype(np.float32); cts = ct["scores"].astype(np.float32)
    fused = (mt - muM) / sdM + beta * (cts - muC) / sdC          # [F,H,W]
    gt = tl[rt["keys"]]                                          # [F,H,W]
    return fused, gt


def ci_of(S, gt_all, n_boot=NBOOT, seed=0):
    F = S.shape[0]
    samp = S[gt_all != 255][::10]
    edges = np.unique(np.quantile(samp.astype(np.float64), np.linspace(0, 1, NBINS + 1)))
    edges[0] -= 1e-6; edges[-1] += 1e-6; nb = len(edges) - 1
    idh = np.zeros((F, nb)); oodh = np.zeros((F, nb))
    for i in range(F):
        s = S[i]; g = gt_all[i]
        idh[i] = np.histogram(s[g == 0], bins=edges)[0]
        oodh[i] = np.histogram(s[g == 1], bins=edges)[0]
    pe = metrics_from_hist(idh.sum(0), oodh.sum(0), edges)
    rng = np.random.default_rng(seed); boot = {"AP": [], "FPR95": []}
    for _ in range(n_boot):
        idx = rng.integers(0, F, F)
        r = metrics_from_hist(idh[idx].sum(0), oodh[idx].sum(0), edges)
        boot["AP"].append(r["AP"]); boot["FPR95"].append(r["FPR95"])
    hw = lambda v: float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2)
    return pe, {"AP": hw(boot["AP"]), "FPR95": hw(boot["FPR95"])}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="all"); a = ap.parse_args()
    DS = DS_ALL if a.dataset == "all" else [a.dataset]
    ref_b1 = {"goose": 0.822, "rellis": 0.588, "wildscenes": 0.744, "rugd": 0.948}   # from tcc_fused
    out = {}
    print(f"{'dataset':11s} {'beta0(MSP-M2F)':>16s} {'beta1 AP[±hw]':>18s} {'beta1 FPR95[±hw]':>18s}  check")
    for ds in DS:
        f0, g = load_fused(ds, 0.0); pe0, _ = ci_of(f0, g, n_boot=1)          # point only for self-check
        del f0
        f1, g = load_fused(ds, 1.0); pe1, hw1 = ci_of(f1, g); del f1
        chk = "" if ds not in ref_b1 else ("OK" if abs(pe1["AP"] - ref_b1[ds]) < 0.01 else f"MISMATCH(tcc_fused={ref_b1[ds]})")
        out[ds] = {"beta0_MSPM2F": pe0, "beta1": pe1, "beta1_ci_halfwidth": hw1}
        print(f"{ds:11s} {pe0['AP']:16.3f} {pe1['AP']:.3f}[{hw1['AP']:.3f}]".ljust(48)
              + f" {pe1['FPR95']:.3f}[{hw1['FPR95']:.3f}]  {chk}")
    json.dump(out, open(f"{R}/ours_score_final.json", "w"), indent=2)
    print("[saved] ours_score_final.json")
