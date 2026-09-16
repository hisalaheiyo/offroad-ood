#!/usr/bin/env python
"""
#3 seed robustness: compare Mask2Former base (RbA/MSP-M2F) + OURS AP/FPR95 at seed0 vs seed1, to show the conclusion is not a single-seed coincidence.
OURS: MSP-M2F_z + β·cDNP_z, β selected by val_AP independently per seed (z-norm uses each seed's own val-ID).
Usage (CPU): python method_seed.py --dataset rellis
"""
import argparse, json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
import metrics as M
R = f"{ROOT}/results"
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

def load(pfx, sp, ssfx, which):
    if which == "cDNP": z = np.load(f"{R}/cdnp_scores_{pfx}{sp}.npz"); return z["scores"].astype(np.float32).ravel(), z["keys"].ravel()
    z = np.load(f"{R}/m2f_rba_scores_{pfx}{sp}{ssfx}.npz"); return z[which].astype(np.float32).ravel(), z["keys"].ravel()

def metrics_seed(dataset, seed):
    pfx = "" if dataset == "goose" else f"{dataset}_"; ssfx = "" if seed == 0 else f"_s{seed}"
    _, ol = get_dataset(dataset); o = ol(); tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    def zp(which):
        st, kt = load(pfx, "test", ssfx, which); sv, kv = load(pfx, "val", ssfx, which)
        idv = tl[kv] == 0; mu = sv[idv].mean(); sd = sv[idv].std() + 1e-6
        return (st - mu) / sd, (sv - mu) / sd, kt, kv
    mt, mv, kt, kv = zp("MSP_M2F"); rt, rv, krt, _ = zp("RbA"); ft, fv, kft, kfv = zp("cDNP")
    # correctness: before fusion, must verify that mask-seed{seed} keys are element-wise aligned with cDNP/RbA (otherwise mt+β·ft is misaligned)
    assert np.array_equal(kt, kft) and np.array_equal(kt, krt), f"{dataset} seed{seed} keys misaligned (mask vs cDNP/RbA)"
    assert np.array_equal(kv, kfv), f"{dataset} seed{seed} val keys misaligned"
    gt = tl[kt]; vt = gt != IGNORE; ybt = (gt[vt] == 1).astype(np.int8)
    gv = tl[kv]; vv = gv != IGNORE; ybv = (gv[vv] == 1).astype(np.int8)
    ev = lambda s, m, y: M.pixel_pr_auroc_fpr95([s[m].astype(np.float16)], [y])
    # select β by val_AP
    bb, ba = 0.0, -1
    for b in BETAS:
        a = ev(mv + b * fv, vv, ybv)["AP"]
        if a > ba: ba, bb = a, b
    return dict(RbA=ev(rt, vt, ybt), MSP_M2F=ev(mt, vt, ybt), OURS=ev(mt + bb * ft, vt, ybt), beta=bb)

def main(dataset="rellis"):
    print(f"\n=== {dataset} seed robustness (seed0 vs seed1) ===")
    r0 = metrics_seed(dataset, 0); r1 = metrics_seed(dataset, 1)
    print(f"{'method':10s} {'AP s0':>7s} {'AP s1':>7s} {'|Δ|':>6s} | {'FPR s0':>7s} {'FPR s1':>7s} {'|Δ|':>6s}")
    for m in ["RbA", "MSP_M2F", "OURS"]:
        dap = abs(r0[m]["AP"] - r1[m]["AP"]); dfp = abs(r0[m]["FPR95"] - r1[m]["FPR95"])
        print(f"{m:10s} {r0[m]['AP']:7.3f} {r1[m]['AP']:7.3f} {dap:6.3f} | {r0[m]['FPR95']:7.3f} {r1[m]['FPR95']:7.3f} {dfp:6.3f}")
    print(f"  β: seed0={r0['beta']} seed1={r1['beta']}")
    json.dump({"seed0": r0, "seed1": r1}, open(f"{R}/method_seed_{'' if dataset=='goose' else dataset+'_'}test.json", "w"), indent=2)
    print("[saved]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="rellis"); a = ap.parse_args(); main(a.dataset)
