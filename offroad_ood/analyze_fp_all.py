#!/usr/bin/env python
"""
Per-method failure analysis (GOOSE test): per-method vegetation false-positive % of the 5 baselines + per-Tier detection recall.
Score sources: dinov2_scores_test.npz (kNN+keys) + seghead_scores_test.npz (MSP/MaxLogit/Energy/SML).
keys serve as the GT source for all methods (same test, same order, same downsample 4). tau = each method's 5% quantile of OOD scores (TPR95 operating point, scale-free).
sanity: the computed global FPR95 should be approximately equal to each method's eval value.
Usage: module load scicomp-python-env/2025.2; python analyze_fp_all.py
"""
import numpy as np
from ontology import load_goose_ontology, IGNORE, ROOT

import json as _json
def _read_eval_fpr95(R, pfx):
    """Dynamically read eval FPR95 from each method's result json for sanity comparison (default if not readable)."""
    e={}
    def g(path, *keys):
        try:
            d=_json.load(open(f"{R}/{path}"))
            for k in keys: d=d[k]
            return float(d)
        except Exception: return None
    e["DINOv2kNN"]=g(f"dinov2_knn_{pfx}test.json","pixel","FPR95")
    for m in ["MSP","MaxLogit","Energy","SML"]: e[m]=g(f"baselines_seghead_{pfx}test.json",m,"FPR95")
    e["cDNP_k5"]=g(f"cdnp_{pfx}test.json","k5","FPR95")
    # maha json: the old goose version uses the "test" key, the new version (--split) uses the "result" key
    e["Mahalanobis"]=g(f"mahalanobis_{pfx}test.json","result","FPR95") or g(f"mahalanobis_{pfx}test.json","test","FPR95")
    return {k:v for k,v in e.items() if v is not None}

def main(dataset="goose"):
    from datasets import get_dataset
    R=f"{ROOT}/results"; pfx="" if dataset=="goose" else f"{dataset}_"
    _,onto_loader=get_dataset(dataset); o=onto_loader(); info=o["info"]
    role=lambda k:info[k]["eff_role"]; name=lambda k:info[k]["name"]; tier=lambda k:info[k]["tier"]
    id_keys=[k for k in info if role(k).startswith("id_")]
    ood_keys=[k for k in info if role(k).startswith("ood_")]
    veg_keys=set(k for k in info if role(k)=="id_vegetation")
    EVAL_FPR95=_read_eval_fpr95(R, pfx)

    dn=np.load(f"{R}/dinov2_scores_{pfx}test.npz"); keys=dn["keys"]
    K=keys.ravel()
    total=np.bincount(K,minlength=256)                     # total pixels per class (full set)
    n_ood=sum(int(total[k]) for k in ood_keys); n_id=sum(int(total[k]) for k in id_keys)
    is_ood_pix = np.isin(K, ood_keys)

    sources={"DINOv2kNN":(dn,"scores")}
    sg=np.load(f"{R}/seghead_scores_{pfx}test.npz")
    for m in ["MSP","MaxLogit","Energy","SML"]: sources[m]=(sg,m)
    for mname,fn in [("cDNP_k5",f"cdnp_scores_{pfx}test.npz"),("Mahalanobis",f"mahalanobis_scores_{pfx}test.npz")]:
        try:
            npz=np.load(f"{R}/{fn}"); sources[mname]=(npz,"scores")
        except FileNotFoundError: pass

    print(f"{'method':11s} {'FPR95(calc)':>9s} {'FPR95(eval)':>11s} {'veg%ofFP':>9s} | per-class top FP")
    rows={}
    for m,(npz,key) in sources.items():
        S=npz[key]
        assert S.shape==keys.shape, f"{m} shape{S.shape}!=keys{keys.shape}"   # alignment assertion
        s=S.astype(np.float32).ravel()
        ood_s=s[is_ood_pix]
        tau=np.percentile(ood_s,5)                          # TPR95: 95% of OOD above tau -> tau = 5% quantile
        above=s>tau
        fp_cnt=np.bincount(K[above],minlength=256)          # per-class pixels > tau (ID class = false positive, OOD class = detection)
        fpr95=sum(int(fp_cnt[k]) for k in id_keys)/n_id     # global false-positive rate (sanity)
        total_fp=sum(int(fp_cnt[k]) for k in id_keys)
        veg_fp=sum(int(fp_cnt[k]) for k in veg_keys)
        veg_share=veg_fp/max(total_fp,1)
        # per-class false-positive share
        cls=sorted([(name(k), role(k)=="id_vegetation", fp_cnt[k]/max(total_fp,1)) for k in id_keys if total[k]>0],
                   key=lambda x:-x[2])[:3]
        top=", ".join(f"{'🌿' if v else ''}{n}{int(s*100)}%" for n,v,s in cls)
        ev=EVAL_FPR95.get(m)
        ok=("✅" if abs(fpr95-ev)<0.03 else "🔴mismatch") if ev is not None else "—"
        evs=f"{ev:11.3f}" if ev is not None else f"{'—':>11s}"
        print(f"{m:11s} {fpr95:9.3f} {evs}{ok} {veg_share*100:8.1f}% | {top}")
        # per-Tier recall
        rec={}
        for t in("T1","T2","T3"):
            tk=[k for k in ood_keys if tier(k)==t]; nt=sum(int(total[k]) for k in tk)
            if nt: rec[t]=sum(int(fp_cnt[k]) for k in tk)/nt
        rows[m]=dict(fpr95=fpr95,veg_share=veg_share,tier_recall=rec)
        del S,s,ood_s,above,fp_cnt
    print("\n=== per-Tier detection recall @ each method's TPR95 operating point ===")
    print(f"{'method':11s} {'T1':>6s} {'T2':>6s} {'T3':>6s}")
    for m,r in rows.items():
        rc=r["tier_recall"]; print(f"{m:11s} {rc.get('T1',0)*100:5.1f}% {rc.get('T2',0)*100:5.1f}% {rc.get('T3',0)*100:5.1f}%")
    fn="failure_analysis.json" if dataset=="goose" else f"failure_analysis_{dataset}.json"
    import json; json.dump(rows,open(f"{R}/{fn}","w"),indent=2)
    print(f"[saved] results/{fn}")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp"]); a=ap.parse_args()
    main(a.dataset)
