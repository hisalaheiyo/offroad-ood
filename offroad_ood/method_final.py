#!/usr/bin/env python
"""
Oracle-beta ablation (validation-selected beta): terrain-distance calibrated mask-transformer OOD.
Upper bound for the fixed-beta=1 "Ours-Score" of ours_score_final.py (the paper's headline detector).
score = MSP-M2F(mask, z) + β·cDNP(terrain-dist, z);  β selected on val (max val AUROC), applied to test.
Motivation (derived from the benchmark analysis): mask has good AP but its high-recall tail admits terrain-FP (poor FPR); cDNP is negative-z on terrain -> adding it suppresses the terrain tail.
No test leakage: z-norm uses val-ID statistics, β selected on val. Report test AP/AUROC/FPR95 vs base (β=0) and the strongest single baseline.
Usage (CPU): python method_final.py --dataset goose [--base MSP-M2F]
"""
import argparse, json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
import metrics as M
R = f"{ROOT}/results"
SRC = {"RbA": ("m2f_rba_scores_{pfx}{sp}.npz", "RbA"), "MSP-M2F": ("m2f_rba_scores_{pfx}{sp}.npz", "MSP_M2F"),
       "cDNP": ("cdnp_scores_{pfx}{sp}.npz", "scores")}
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

def get(ds, name, sp):
    pfx = "" if ds == "goose" else f"{ds}_"; fn, key = SRC[name]
    z = np.load(f"{R}/{fn.format(pfx=pfx, sp=sp)}")
    return z[key].astype(np.float32).ravel(), z["keys"].ravel()

def zpair(ds, name):
    """Returns (z_test, z_val, keys_test, keys_val), z-norm uses val-ID."""
    st, kt = get(ds, name, "test"); sv, kv = get(ds, name, "val")
    _, ol = get_dataset(ds); o = ol(); tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    idv = tl[kv] == 0; mu = sv[idv].mean(); sd = sv[idv].std() + 1e-6
    return (st - mu) / sd, (sv - mu) / sd, kt, kv, tl

def main(dataset="goose", base="MSP-M2F"):
    bt, bv, kt, kv, tl = zpair(dataset, base)
    ft, fv, kt2, kv2, _ = zpair(dataset, "cDNP")
    assert np.array_equal(kt, kt2) and np.array_equal(kv, kv2), "keys misaligned"
    gtt = tl[kt]; vt = gtt != IGNORE; ybt = (gtt[vt] == 1).astype(np.int8)
    gtv = tl[kv]; vv = gtv != IGNORE; ybv = (gtv[vv] == 1).astype(np.int8)
    def ev(s, mask, yb): return M.pixel_pr_auroc_fpr95([s[mask].astype(np.float16)], [yb])
    # select β on val (max val AP — primary metric, standard practice; val_AUROC unreliable, see method_sweep)
    best_b, best_va = 0.0, -1
    for b in BETAS:
        va = ev(bv + b * fv, vv, ybv)["AP"]
        if va > best_va: best_va, best_b = va, b
    # test report
    base_te = ev(bt, vt, ybt)
    our_te = ev(bt + best_b * ft, vt, ybt)
    # strongest single baseline (base vs cDNP vs RbA) on test
    singles = {}
    for nm in [base, "cDNP", "RbA"]:
        z, _, k, _, _ = zpair(dataset, nm)
        if np.array_equal(k, kt): singles[nm] = ev(z, vt, ybt)
    bs = max(singles, key=lambda m: singles[m]["AP"])
    print(f"\n=== {dataset} Terrain-Calibrated {base} (val-selected β*={best_b}) ===")
    print(f"{'':16s} {'AP':>7s} {'AUROC':>7s} {'FPR95':>7s}")
    print(f"{'base('+base+')':16s} {base_te['AP']:7.3f} {base_te['AUROC']:7.3f} {base_te['FPR95']:7.3f}")
    print(f"{'best-single('+bs+')':16s} {singles[bs]['AP']:7.3f} {singles[bs]['AUROC']:7.3f} {singles[bs]['FPR95']:7.3f}")
    print(f"{'OURS(β='+str(best_b)+')':16s} {our_te['AP']:7.3f} {our_te['AUROC']:7.3f} {our_te['FPR95']:7.3f}")
    dAP = our_te['AP'] - singles[bs]['AP']; dF = our_te['FPR95'] - singles[bs]['FPR95']
    print(f"  Δvs best-single: AP{dAP:+.3f} FPR95{dF:+.3f}")
    out = dict(base=base, beta=best_b, base_test=base_te, best_single=bs, best_single_test=singles[bs], ours_test=our_te)
    json.dump(out, open(f"{R}/method_final_{base}_{'' if dataset=='goose' else dataset+'_'}test.json", "w"), indent=2)
    print("[saved]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose"); ap.add_argument("--base", default="MSP-M2F")
    a = ap.parse_args(); main(a.dataset, a.base)
