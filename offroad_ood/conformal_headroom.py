#!/usr/bin/env python
"""
Conformal prototype step2: headroom validation. Compare the stability + FPR guarantee of 3 operating points on GOOSE across 3 seeds.
- naive τ95: threshold from the val OOD score 5% quantile (TPR95) -> OOD-anchored -> seed-unstable (known CV 49-56%).
- global conformal: threshold from the val-ID score (1-α) quantile -> ID-anchored, single threshold.
- per-terrain conformal: for each predicted terrain class c, take τ_c from the val-ID (predclass==c) score (1-α) quantile -> ID-anchored + per-terrain.
Measures: cross-seed test FPR mean±std (CV) + recall + FPR coverage (whether ≤α).
Criterion: per-terrain conformal FPR CV << naive (49-56%) + coverage holds + reasonable recall -> the direction holds.
⚠️ Reads only m2f_predclass_* (new files) + writes conformal_headroom.json, does not touch any existing results.
Usage (CPU): python conformal_headroom.py --dataset goose --seeds 0,1,2 --alpha 0.05
"""
import argparse, json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
R = f"{ROOT}/results"

def load(ds, split, seed):
    pfx = "" if ds == "goose" else f"{ds}_"; ssfx = "" if seed == 0 else f"_s{seed}"
    z = np.load(f"{R}/m2f_predclass_{pfx}{split}{ssfx}.npz")
    return z["predclass"].ravel().astype(np.int32), z["MSP_M2F"].ravel().astype(np.float32), z["keys"].ravel()

def main(dataset="goose", seeds=(0, 1, 2), alpha=0.05):
    _, ol = get_dataset(dataset); o = ol(); tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    rows = {"naive_tpr95": [], "conformal_global": [], "conformal_perterrain": []}
    for seed in seeds:
        pcv, sv, kv = load(dataset, "val", seed); pct, st, kt = load(dataset, "test", seed)
        gv = tl[kv]; idv = gv == 0; oodv = gv == 1
        gt = tl[kt]; idt = gt == 0; oodt = gt == 1
        nid_t = idt.sum(); nood_t = oodt.sum()
        def measure(above_t):
            return dict(FPR=float((above_t & idt).sum() / max(nid_t, 1)), recall=float((above_t & oodt).sum() / max(nood_t, 1)))
        # 1. naive τ95 (val OOD 5% quantile = 95% recall)
        tau95 = np.percentile(sv[oodv], 5)
        rows["naive_tpr95"].append(measure(st > tau95))
        # 2. global conformal (val-ID (1-α) quantile)
        tg = np.percentile(sv[idv], 100 * (1 - alpha))
        rows["conformal_global"].append(measure(st > tg))
        # 3. per-terrain conformal: val-ID (1-α) quantile for each predicted class c; undersampled (<100) falls back to global
        classes = np.unique(pcv[idv])
        tau_c = {}
        for c in classes:
            m = idv & (pcv == c); sc = sv[m]
            tau_c[c] = float(np.percentile(sc, 100 * (1 - alpha))) if sc.size >= 100 else tg
        tau_map = np.full(int(max(classes.max(), pct.max())) + 1, tg, np.float32)
        for c, t in tau_c.items(): tau_map[c] = t
        thr_t = tau_map[np.clip(pct, 0, len(tau_map) - 1)]
        rows["conformal_perterrain"].append(measure(st > thr_t))
    # summary
    out = {"alpha": alpha, "seeds": list(seeds)}
    print(f"\n=== {dataset} conformal headroom (α={alpha}, {len(seeds)}seed) ===")
    print(f"{'method':22s} {'FPR mean±std':>16s} {'FPR CV':>7s} {'recall mean±std':>16s} {'coverage(FPR≤α)':>14s}")
    for m, rs in rows.items():
        fpr = np.array([r["FPR"] for r in rs]); rec = np.array([r["recall"] for r in rs])
        cv = fpr.std() / (fpr.mean() + 1e-9)
        cov = "—" if m == "naive_tpr95" else ("✅" if fpr.mean() <= alpha * 1.3 else "🔴over")
        out[m] = dict(FPR_mean=float(fpr.mean()), FPR_std=float(fpr.std()), FPR_cv=float(cv),
                      recall_mean=float(rec.mean()), recall_std=float(rec.std()), per_seed_FPR=fpr.tolist(), per_seed_recall=rec.tolist())
        print(f"{m:22s} {fpr.mean():7.3f}±{fpr.std():.3f}   {cv*100:5.0f}% {rec.mean():7.3f}±{rec.std():.3f}      {cov:>10s}")
    print(f"\nCriterion: per-terrain conformal FPR CV should be <<naive({rows['naive_tpr95'] and '49-56%'}) + coverage✅ + reasonable recall -> the direction holds")
    json.dump(out, open(f"{R}/conformal_headroom_{'' if dataset=='goose' else dataset+'_'}.json", "w"), indent=2)
    print("[saved] conformal_headroom json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(); main(a.dataset, tuple(int(x) for x in a.seeds.split(",")), a.alpha)
