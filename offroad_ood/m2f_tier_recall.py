#!/usr/bin/env python
"""
deep-dive: does the mask-transformer miss small anomalous objects? Decompose per-Tier pixel recall at the overall TPR95 operating point.
Hypothesis: mask relies on grouping into object masks -> big help for large objects (T1 vehicles/buildings), small objects (T2 debris) may be missed by queries -> T2 recall << T1.
Compare mask (RbA/MSP-M2F) vs per-pixel (cDNP/kNN). If mask T2 recall is clearly lower than per-pixel -> a key robotics limitation (small debris missed).
Usage: python m2f_tier_recall.py
"""
import json, numpy as np
from ontology import ROOT
from datasets import get_dataset
R = f"{ROOT}/results"
DSS = ["goose", "rellis", "wildscenes", "rugd"]
SRC = {"RbA": ("m2f_rba_scores_{pfx}test.npz", "RbA"), "MSP-M2F": ("m2f_rba_scores_{pfx}test.npz", "MSP_M2F"),
       "cDNP": ("cdnp_scores_{pfx}test.npz", "scores"), "kNN": ("dinov2_scores_{pfx}test.npz", "scores")}

def main():
    out = {}
    for ds in DSS:
        pfx = "" if ds == "goose" else f"{ds}_"
        _, ol = get_dataset(ds); o = ol(); info = o["info"]
        tier_of = {k: info[k]["tier"] for k in o["ood_keys"]}
        tiers = sorted(set(t for t in tier_of.values() if t))
        out[ds] = {}
        print(f"\n=== {ds} per-Tier pixel recall @overall TPR95 ===")
        print(f"{'method':10s} " + " ".join(f"{t:>7s}" for t in tiers) + f"  {'(T2/T1)':>8s}")
        for m, (fn, key) in SRC.items():
            try:
                z = np.load(f"{R}/{fn.format(pfx=pfx)}"); s = z[key].astype(np.float32).ravel(); K = z["keys"].ravel()
            except (FileNotFoundError, KeyError):
                continue
            # tier LUT: label_key -> tier index (0..len-1), non-OOD=-1; one mapping to avoid repeated np.isin
            tlut = np.full(256, -1, np.int8)
            for k in o["ood_keys"]:
                if tier_of[k] in tiers: tlut[k] = tiers.index(tier_of[k])
            tarr = tlut[K]
            is_ood = tarr >= 0
            tau = np.percentile(s[is_ood], 5)              # overall TPR95
            above = s > tau
            rec = {}
            for ti, t in enumerate(tiers):
                mask = tarr == ti; n = mask.sum()
                rec[t] = float((above & mask).sum() / max(n, 1))
            ratio = rec.get("T2", 0) / max(rec.get("T1", 1e-9), 1e-9)
            out[ds][m] = dict(tier_recall=rec, t2_over_t1=ratio)
            print(f"{m:10s} " + " ".join(f"{rec[t]*100:6.1f}%" for t in tiers) + f"  {ratio:8.2f}")
    json.dump(out, open(f"{R}/m2f_tier_recall.json", "w"), indent=2)
    print("\n[saved] m2f_tier_recall.json")
    print("Interpretation: if mask T2/T1 << per-pixel T2/T1 -> mask misses small objects (robotics limitation); if similar -> mask does not sacrifice small objects")

if __name__ == "__main__": main()
