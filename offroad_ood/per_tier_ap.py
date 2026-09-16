#!/usr/bin/env python
"""
Per-tier AP, to check whether "T2 is hardest" holds (recall does not support it, look at AP).
Per tier: positive class = OOD pixels of that tier, negative class = ID pixels, everything else (other-tier OOD + ignore) discarded → AP.
Tier definitions reuse m2f_tier_recall (o["info"][k]["tier"]). Methods use the directly cached MSP-M2F/RbA/cDNP (no fusion, low risk).
⚠️ Read cached only + write new file per_tier_ap.json, does not touch existing results.
Self-check: overall AP merging all tiers (pos = any OOD) should = that method's AP in the main table → validates the tier partition is correct.
Usage (CPU): python per_tier_ap.py --dataset rugd
"""
import argparse, json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
import metrics as M
R = f"{ROOT}/results"
SRC = {"cDNP": ("cdnp_scores_{pfx}test.npz", "scores"),
       "RbA": ("m2f_rba_scores_{pfx}test.npz", "RbA"),
       "MSP-M2F": ("m2f_rba_scores_{pfx}test.npz", "MSP_M2F")}

def ap_of(s, pos, neg):
    """AP: positive class pos, negative class neg (boolean, same length)."""
    m = pos | neg
    y = pos[m].astype(np.int8)
    return M.pixel_pr_auroc_fpr95([s[m].astype(np.float16)], [y])["AP"]

def main(dataset="rugd"):
    pfx = "" if dataset == "goose" else f"{dataset}_"
    _, ol = get_dataset(dataset); o = ol(); info = o["info"]
    tl = np.full(256, IGNORE, np.uint8)
    for k, v in o["test_lut"].items(): tl[k] = v
    tier_of = {k: info[k]["tier"] for k in o["ood_keys"]}
    tiers = sorted(set(t for t in tier_of.values() if t))
    tlut = np.full(256, -1, np.int8)
    for k in o["ood_keys"]:
        if tier_of[k] in tiers: tlut[k] = tiers.index(tier_of[k])
    print(f"=== {dataset} per-Tier AP (tiers={tiers}) ===", flush=True)
    out = {"tiers": tiers, "methods": {}}
    for m, (fn, key) in SRC.items():
        try:
            z = np.load(f"{R}/{fn.format(pfx=pfx)}"); s = z[key].astype(np.float32).ravel(); K = z["keys"].ravel()
        except (FileNotFoundError, KeyError):
            print(f"{m}: scores missing, skip"); continue
        idm = tl[K] == 0                       # ID negative class
        tarr = tlut[K]
        # self-check: overall AP (pos = any OOD)
        overall = ap_of(s, tarr >= 0, idm)
        row = {"overall_AP": float(overall), "per_tier": {}, "npix": {}}
        for ti, t in enumerate(tiers):
            pos = tarr == ti; n = int(pos.sum())
            row["npix"][t] = n
            row["per_tier"][t] = float(ap_of(s, pos, idm)) if n >= 100 else None
        out["methods"][m] = row
        ts = " ".join(f"{t}={row['per_tier'][t]:.3f}" if row['per_tier'][t] is not None else f"{t}=--" for t in tiers)
        npx = " ".join(f"{t}:{row['npix'][t]}" for t in tiers)
        print(f"{m:10s} overall_AP={overall:.3f} (self-check should = main table) | {ts} | npix {npx}", flush=True)
    # --- Ours-Score (fixed beta=1 fusion of MSP-M2F + cDNP), same tier protocol ---
    try:
        rt = np.load(f"{R}/m2f_rba_scores_{pfx}test.npz"); ct = np.load(f"{R}/cdnp_scores_{pfx}test.npz")
        rv = np.load(f"{R}/m2f_rba_scores_{pfx}val.npz"); cv = np.load(f"{R}/cdnp_scores_{pfx}val.npz")
        assert np.array_equal(rt["keys"], ct["keys"]), "test keys misaligned (rba vs cdnp)"
        Kf = rt["keys"].ravel()
        mvv = rv["MSP_M2F"].astype(np.float32).ravel(); cvv = cv["scores"].astype(np.float32).ravel()
        gvv = tl[rv["keys"].ravel()] == 0
        muM, sdM = mvv[gvv].mean(), mvv[gvv].std() + 1e-6
        muC, sdC = cvv[gvv].mean(), cvv[gvv].std() + 1e-6
        fs = (rt["MSP_M2F"].astype(np.float32).ravel() - muM) / sdM + (ct["scores"].astype(np.float32).ravel() - muC) / sdC
        idm = tl[Kf] == 0; tarr = tlut[Kf]
        overall = ap_of(fs, tarr >= 0, idm)
        row = {"overall_AP": float(overall), "per_tier": {}, "npix": {}}
        for ti, t in enumerate(tiers):
            pos = tarr == ti; n = int(pos.sum()); row["npix"][t] = n
            row["per_tier"][t] = float(ap_of(fs, pos, idm)) if n >= 100 else None
        out["methods"]["Ours-Score"] = row
        ts = " ".join(f"{t}={row['per_tier'][t]:.3f}" if row['per_tier'][t] is not None else f"{t}=--" for t in tiers)
        print(f"{'Ours-Score':10s} overall_AP={overall:.3f} | {ts}", flush=True)
    except Exception as e:
        print(f"Ours-Score skip: {e}", flush=True)
    json.dump(out, open(f"{R}/per_tier_ap_{pfx}test.json", "w"), indent=2)
    print(f"[saved] per_tier_ap_{pfx}test.json", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="rugd"); a = ap.parse_args()
    main(a.dataset)
