#!/usr/bin/env python
"""
Object-level F1* for the fused score (OURS), computed with the same pipeline as the other methods for comparability.
OURS score = z(MSP-M2F)_test + β·z(cDNP)_test (val-ID normalization; β selected by val-AP from BETAS, replicating method_ci).
F1* = component_f1_aggregate averaged over test-OOD quantile thresholds (TPR95..TPR30, 9 points), exactly following m2f_component_f1.py.
⚠️ Read cached scores only + write new file ours_compf1_*.json, does not touch any existing results.
Self-check: (1) fused pixel-AP should ≈ method_ci's OURS AP; (2) this script's recomputed MSP-M2F F1* should ≈ existing summary compf1 value (validates the pipeline).
Usage (CPU): python ours_component_f1.py --dataset rellis
"""
import argparse, json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
import metrics as M
R = f"{ROOT}/results"
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # consistent with method_ci
SRC = {"MSP-M2F": ("m2f_rba_scores_{pfx}{sp}.npz", "MSP_M2F"),
       "cDNP": ("cdnp_scores_{pfx}{sp}.npz", "scores")}

def load(ds, name, sp):
    pfx = "" if ds == "goose" else f"{ds}_"; fn, key = SRC[name]
    z = np.load(f"{R}/{fn.format(pfx=pfx, sp=sp)}")
    return z[key].astype(np.float32), z["keys"]

def znorm(ds, name, tl):
    """z-norm by val-ID (replicating method_ci.zt): returns test(F,h,w), val(F,h,w), Kt, Kv."""
    St, Kt = load(ds, name, "test"); Sv, Kv = load(ds, name, "val")
    idv = tl[Kv].ravel() == 0
    mu = float(Sv.ravel()[idv].mean()); sd = float(Sv.ravel()[idv].std() + 1e-6)
    return (St - mu) / sd, (Sv - mu) / sd, Kt, Kv

def f1star(score_frames, gt_frames):
    """F1* = component-F1 averaged over test-OOD quantile thresholds (copied from m2f_component_f1)."""
    tood = np.concatenate([s.ravel() for s in score_frames])[np.concatenate([g.ravel() for g in gt_frames]) == 1]
    threshs = np.percentile(tood, np.linspace(5, 70, 9))
    f1s = [M.component_f1_aggregate(score_frames, gt_frames, float(t))["component_F1"] for t in threshs]
    return float(np.mean(f1s)), f1s

def main(dataset="rellis"):
    _, ol = get_dataset(dataset); o = ol(); tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    bt, bv, Kt, Kv = znorm(dataset, "MSP-M2F", tl)
    ft, fv, Kt2, Kv2 = znorm(dataset, "cDNP", tl)
    assert np.array_equal(Kt, Kt2) and np.array_equal(Kv, Kv2), "keys not aligned!"
    gt_all = tl[Kt]                                   # (F,h,w) test GT 0/1/255
    # β selected by val-AP (replicating method_ci: direct AP, not histogram)
    gv = tl[Kv]; vv = gv.ravel() != IGNORE; ybv = (gv.ravel()[vv] == 1).astype(np.int8)
    def valap(sf): return M.pixel_pr_auroc_fpr95([sf.ravel()[vv].astype(np.float16)], [ybv])["AP"]
    best_b, ba = 0.0, -1.0
    for b in BETAS:
        a = valap(bv + b * fv)
        if a > ba: ba, best_b = a, b
    print(f"[{dataset}] β*={best_b} (val_AP={ba:.4f})", flush=True)
    ours = bt + best_b * ft                           # (F,h,w) OURS test score
    # self-check 1: fused pixel-AP (test) should ≈ method_ci's OURS AP
    vt = gt_all.ravel() != IGNORE; ybt = (gt_all.ravel()[vt] == 1).astype(np.int8)
    ap_ours = M.pixel_pr_auroc_fpr95([ours.ravel()[vt].astype(np.float16)], [ybt])["AP"]
    print(f"  [self-check 1] fused test pixel-AP = {ap_ours:.4f}  (should ≈ main-table OURS AP)", flush=True)
    # frame list
    gt_frames = [gt_all[i] for i in range(gt_all.shape[0])]
    ours_frames = [ours[i] for i in range(ours.shape[0])]
    msp_frames = [bt[i] for i in range(bt.shape[0])]
    # self-check 2: this script's recomputed MSP-M2F F1* → compare against existing summary compf1
    f1_msp, _ = f1star(msp_frames, gt_frames)
    f1_ours, per = f1star(ours_frames, gt_frames)
    print(f"  [self-check 2] MSP-M2F F1* (this script) = {f1_msp:.4f}  (should ≈ existing summary compf1 MSP-M2F)", flush=True)
    print(f"  OURS F1* = {f1_ours:.4f}", flush=True)
    out = {"beta": best_b, "fused_pixel_AP": ap_ours,
           "OURS": {"component_F1": f1_ours, "per_thresh": per, "paradigm": "mask"},
           "MSP-M2F_recheck": f1_msp}
    pfx = "" if dataset == "goose" else dataset + "_"
    json.dump(out, open(f"{R}/ours_compf1_{pfx}test.json", "w"), indent=2)
    print(f"[saved] ours_compf1_{pfx}test.json", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="rellis"); a = ap.parse_args()
    main(a.dataset)
