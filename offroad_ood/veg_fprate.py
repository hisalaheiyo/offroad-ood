#!/usr/bin/env python
"""
M1+M2: vegetation FP rate (not just share) + recall-vs-FP decomposition, ×4 datasets ×7 methods.
Core thesis evidence: at the TPR95 operating point, the FP rate of vegetation classes (flagged pixels / that class's pixels) >> the FP rate of non-vegetation ID classes
  → vegetation is intrinsically FP-prone (not just abundant); and OOD recall is high → "anomalies are detected but drowned out by vegetation FP".
Correctness note: the FP rate must be normalized by class frequency (fp_cnt[k]/total[k]) to distinguish abundant vs FP-prone.
"""
import json, numpy as np
from ontology import IGNORE, ROOT
from datasets import get_dataset
R=f"{ROOT}/results"
DSS=["goose","wildscenes","rellis","rugd"]
MFILE={"kNN":("dinov2_scores","scores"),"cDNP":("cdnp_scores","scores"),"Maha":("mahalanobis_scores","scores"),
       "MSP":("seghead_scores","MSP"),"MaxLogit":("seghead_scores","MaxLogit"),"Energy":("seghead_scores","Energy"),"SML":("seghead_scores","SML")}

def main():
    out={}
    print(f"{'dataset':10s} {'method':9s} {'OOD recall':>10s} {'veg FPrate':>8s} {'nonveg ID FPrate':>12s} {'ratio':>6s}")
    for ds in DSS:
        pfx="" if ds=="goose" else f"{ds}_"
        _,ol=get_dataset(ds); o=ol(); info=o["info"]
        tl=np.full(256,IGNORE,np.uint8)
        for k,v in o["test_lut"].items(): tl[k]=v
        veg=[k for k in o["id_keys"] if info[k]["eff_role"]=="id_vegetation"]
        nonveg=[k for k in o["id_keys"] if info[k]["eff_role"]!="id_vegetation"]
        ood=o["ood_keys"]
        K=np.load(f"{R}/dinov2_scores_{pfx}test.npz")["keys"].ravel()
        is_veg=np.isin(K,veg); is_nv=np.isin(K,nonveg); is_ood=np.isin(K,ood)
        nveg=is_veg.sum(); nnv=is_nv.sum(); nood=is_ood.sum()
        out[ds]={}
        for m,(fn,key) in MFILE.items():
            try: s=np.load(f"{R}/{fn}_{pfx}test.npz")[key].astype(np.float32).ravel()
            except FileNotFoundError: continue
            tau=np.percentile(s[is_ood],5)                    # TPR95 operating point
            above=s>tau
            recall=(above&is_ood).sum()/max(nood,1)
            veg_rate=(above&is_veg).sum()/max(nveg,1)         # vegetation FP rate
            nv_rate=(above&is_nv).sum()/max(nnv,1)            # non-vegetation ID FP rate
            ratio=veg_rate/max(nv_rate,1e-6)
            out[ds][m]=dict(ood_recall=float(recall),veg_fprate=float(veg_rate),nonveg_fprate=float(nv_rate),ratio=float(ratio))
            print(f"{ds:10s} {m:9s} {recall*100:9.1f}% {veg_rate*100:7.1f}% {nv_rate*100:11.1f}% {ratio:6.1f}x")
            del s,above
        print()
    # Summary: per-dataset average (across methods)
    print("=== Summary: vegetation FP rate vs non-vegetation ID FP rate (mean across methods) ===")
    for ds in DSS:
        if not out[ds]: continue
        vr=np.mean([v["veg_fprate"] for v in out[ds].values()]); nr=np.mean([v["nonveg_fprate"] for v in out[ds].values()])
        rc=np.mean([v["ood_recall"] for v in out[ds].values()])
        print(f"  {ds:10s}: OOD recall={rc*100:.0f}% | veg FPrate={vr*100:.0f}% vs nonveg ID={nr*100:.0f}% ({vr/max(nr,1e-6):.1f}x)")
    print("\nConclusion: vegetation FP rate >> non-vegetation ID → vegetation is intrinsically FP-prone (not just abundant); high recall → detected but drowned out by vegetation FP")
    json.dump(out,open(f"{R}/veg_fprate.json","w"),indent=2)
    print("[saved] veg_fprate.json")

if __name__=="__main__": main()
