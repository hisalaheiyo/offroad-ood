#!/usr/bin/env python
"""
Method CI: bootstrap 95% CI of OURS (=MSP-M2F_z + β·cDNP_z, β selected by val_AP) + paired delta CI (OURS − base) to show the improvement is significant.
Frame-level histogram bootstrap (same as the main-table compute_ci). paired: resample frames, compute OURS and base on the same frames and take the delta.
Usage (CPU batch): python method_ci.py --dataset goose
"""
import argparse, json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
from compute_ci import metrics_from_hist, NBINS, NBOOT
R = f"{ROOT}/results"
SRC = {"MSP-M2F": ("m2f_rba_scores_{pfx}{sp}.npz", "MSP_M2F"), "RbA": ("m2f_rba_scores_{pfx}{sp}.npz", "RbA"),
       "cDNP": ("cdnp_scores_{pfx}{sp}.npz", "scores")}
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

def load(ds, name, sp):
    pfx = "" if ds == "goose" else f"{ds}_"; fn, key = SRC[name]
    z = np.load(f"{R}/{fn.format(pfx=pfx, sp=sp)}"); return z[key].astype(np.float32), z["keys"]

def hists(S, gt_all, edges):
    """per-frame ID/OOD histograms. S,gt_all: (F,h,w)."""
    F = S.shape[0]; nb = len(edges) - 1
    idh = np.zeros((F, nb)); oodh = np.zeros((F, nb))
    for i in range(F):
        s = S[i]; g = gt_all[i]
        idh[i] = np.histogram(s[g == 0], bins=edges)[0]
        oodh[i] = np.histogram(s[g == 1], bins=edges)[0]
    return idh, oodh

def main(dataset="goose"):
    _, ol = get_dataset(dataset); o = ol(); tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    def zt(name):  # z-norm by val-ID, returns test (F,h,w) + val (F,h,w)
        St, Kt = load(dataset, name, "test"); Sv, Kv = load(dataset, name, "val")
        idv = tl[Kv].ravel() == 0; vflat = Sv.ravel()[idv]
        mu = vflat.mean(); sd = vflat.std() + 1e-6
        return (St - mu) / sd, (Sv - mu) / sd, Kt, Kv
    bt, bv, Kt, Kv = zt("MSP-M2F"); ft, fv, _, _ = zt("cDNP")
    rt, rv, Kr, _ = zt("RbA")
    gt_all = tl[Kt]                                        # (F,h,w)
    # select β on val (max val AP)
    gv = tl[Kv]; vv = gv.ravel() != IGNORE; ybv = (gv.ravel()[vv] == 1).astype(np.int8)
    import metrics as M
    def evflat(sf): return M.pixel_pr_auroc_fpr95([sf.ravel()[vv].astype(np.float16)], [ybv])
    best_b, ba = 0.0, -1
    for b in BETAS:
        a = evflat(bv + b * fv)["AP"]
        if a > ba: ba, best_b = a, b
    print(f"[{dataset}] val_AP selected β*={best_b}", flush=True)
    # test scores of the three methods (F,h,w)
    methods = {"base(MSP-M2F)": bt, "RbA": rt, f"OURS(β={best_b})": bt + best_b * ft}
    rng = np.random.default_rng(0); Fn = bt.shape[0]; out = {}
    # pre-build histograms for each method
    H = {}
    for nm, S in methods.items():
        samp = S[gt_all != 255][::10]
        edges = np.unique(np.quantile(samp.astype(np.float64), np.linspace(0, 1, NBINS + 1)))
        edges[0] -= 1e-6; edges[-1] += 1e-6
        H[nm] = (hists(S, gt_all, edges), edges)
    # point estimate + bootstrap (share the same frame resampling -> paired)
    idxs = [rng.integers(0, Fn, Fn) for _ in range(NBOOT)]
    def boot_metric(nm):
        (idh, oodh), edges = H[nm]
        pe = metrics_from_hist(idh.sum(0), oodh.sum(0), edges)
        bt_ = {"AP": [], "AUROC": [], "FPR95": []}
        for idx in idxs:
            r = metrics_from_hist(idh[idx].sum(0), oodh[idx].sum(0), edges)
            for k in bt_: bt_[k].append(r[k])
        return pe, bt_
    ci = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    print(f"{'method':16s} {'AP[CI]':>20s} {'AUROC[CI]':>20s} {'FPR95[CI]':>20s}")
    boots = {}
    for nm in methods:
        pe, bt_ = boot_metric(nm); boots[nm] = bt_
        out[nm] = dict(point=pe, ci={k: ci(bt_[k]) for k in bt_})
        f = lambda k: f"{np.mean(bt_[k]):.3f}[{np.percentile(bt_[k],2.5):.3f},{np.percentile(bt_[k],97.5):.3f}]"
        print(f"{nm:16s} {f('AP'):>20s} {f('AUROC'):>20s} {f('FPR95'):>20s}")
    # paired delta CI: OURS − base (same-frame resampling)
    ours = [k for k in methods if k.startswith("OURS")][0]
    for k in ["AP", "FPR95"]:
        d = np.array(boots[ours][k]) - np.array(boots["base(MSP-M2F)"][k])
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        sig = "✅significant" if (lo > 0 or hi < 0) else "not significant"
        out[f"delta_{k}_vs_base"] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi))
        print(f"  Δ{k}(OURS−base): {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")
    # vs RbA
    for k in ["AP", "FPR95"]:
        d = np.array(boots[ours][k]) - np.array(boots["RbA"][k])
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        sig = "✅significant" if (lo > 0 or hi < 0) else "not significant"
        out[f"delta_{k}_vs_RbA"] = dict(mean=float(d.mean()), lo=float(lo), hi=float(hi))
        print(f"  Δ{k}(OURS−RbA):  {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")
    json.dump(out, open(f"{R}/method_ci_{'' if dataset=='goose' else dataset+'_'}test.json", "w"), indent=2)
    print("[saved]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose"); a = ap.parse_args(); main(a.dataset)
