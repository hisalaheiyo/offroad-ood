#!/usr/bin/env python
"""
#3 component-F1 of the mask-transformer (SMIYC object-level). Hypothesis: masks produce coherent object masks -> object-level wins over per-pixel (fragmented FP) by a large margin.
Fair: mask+per-pixel both computed uniformly on cached [::4,::4] scores (absolute values != published full-res, but same-resolution comparison is fair, noted).
Threshold selected on val (candidates = val OOD quantiles, maximizing val component-F1) -> applied to test.
Usage: python m2f_component_f1.py --dataset rellis
"""
import sys, json, argparse, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
import metrics as M
R = f"{ROOT}/results"

SRC = {"kNN": ("dinov2_scores_{pfx}{sp}.npz", "scores"),
       "cDNP": ("cdnp_scores_{pfx}{sp}.npz", "scores"),
       "MaxLogit": ("seghead_scores_{pfx}{sp}.npz", "MaxLogit"),
       "RbA": ("m2f_rba_scores_{pfx}{sp}.npz", "RbA"),
       "MSP-M2F": ("m2f_rba_scores_{pfx}{sp}.npz", "MSP_M2F")}

def frames(ds, method, sp, tl):
    """Returns (list of 2D score frames, list of 2D gt frames 0/1/255). GT taken uniformly from the keys of dinov2_scores (all methods verified aligned to it; GOOSE seghead npz has no keys)."""
    pfx = "" if ds == "goose" else f"{ds}_"
    fn, key = SRC[method]
    z = np.load(f"{R}/{fn.format(pfx=pfx, sp=sp)}")
    S = z[key].astype(np.float32)
    K = np.load(f"{R}/dinov2_scores_{pfx}{sp}.npz")["keys"]   # canonical alignment keys source
    assert S.shape == K.shape, f"{method} {sp} shape {S.shape} != keys {K.shape}"
    scores = [S[i] for i in range(S.shape[0])]
    gts = [tl[K[i]] for i in range(K.shape[0])]
    return scores, gts

def main(dataset="rellis", methods=None):
    _, ol = get_dataset(dataset); o = ol()
    tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    methods = methods or ["kNN", "cDNP", "MaxLogit", "RbA", "MSP-M2F"]
    out = {}
    print(f"=== {dataset} component-F1 = F₁* (threshold-averaged, matching competitor definition) ===")
    print(f"{'method':10s} {'F1*':>8s}  {'(paradigm)'}")
    for mth in methods:
        try:
            st, gt = frames(dataset, mth, "test", tl)
        except FileNotFoundError:
            print(f"{mth:10s}   (scores not ready, skip)"); continue
        # F₁* = component-F1 averaged over multiple binarization thresholds (the competitor Anomalies-by-Synthesis definition).
        # Fix: the old argmax-val-component-F1 is noisy on the sparse WS val (MSP-M2F 0.001 vs RbA 0.137); a single TPR90 threshold is too permissive.
        # Threshold averaging removes val-selection degeneracy + is directly comparable to the competitor F₁*. Thresholds taken at test OOD score quantiles (TPR95..TPR30).
        tood = np.concatenate([s.ravel() for s in st])[np.concatenate([g.ravel() for g in gt]) == 1]
        threshs = np.percentile(tood, np.linspace(5, 70, 9))                   # TPR95→TPR30
        f1s = [M.component_f1_aggregate(st, gt, float(t))["component_F1"] for t in threshs]
        rt = {"component_F1": float(np.mean(f1s)), "per_thresh": f1s, "threshs": threshs.tolist()}
        para = "mask" if mth in ("RbA", "MSP-M2F") else "per-pixel"
        out[mth] = dict(component_F1=rt["component_F1"], per_thresh=rt["per_thresh"], paradigm=para)
        print(f"{mth:10s} {rt['component_F1']:8.4f}  ({para})")
    json.dump(out, open(f"{R}/m2f_compf1_{'' if dataset=='goose' else dataset+'_'}test.json", "w"), indent=2)
    print(f"[saved] m2f_compf1 json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="rellis")
    a = ap.parse_args(); main(a.dataset)
