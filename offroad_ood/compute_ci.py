#!/usr/bin/env python
"""
7-method main table bootstrap 95% CI (frame-level resampling).
Efficient + correct: per frame, precompute (ID score histogram, OOD score histogram); bootstrap resample frames -> accumulate histograms -> compute AP/AUROC/FPR95 from histograms.
No pixel downsampling (preserves the true ID/OOD ratio). Self-check: the histogram-method point estimate must be approximately equal to the exact pixel_pr value.
Usage (CPU node): python compute_ci.py
"""
import json, numpy as np
from ontology import load_goose_ontology, IGNORE, ROOT
import metrics as M

NBINS=4000; NBOOT=1000

def metrics_from_hist(idh, oodh, edges):
    """Compute AP/AUROC/FPR95 from ID/OOD score histograms (accumulate from high to low scores)."""
    P=oodh.sum(); N=idh.sum()
    if P==0 or N==0: return dict(AP=np.nan,AUROC=np.nan,FPR95=np.nan)
    tp=np.cumsum(oodh[::-1])[::-1]            # OOD >= bin (TP)
    fp=np.cumsum(idh[::-1])[::-1]             # ID >= bin (FP)
    tpr=tp/P; fpr=fp/N
    prec=tp/np.maximum(tp+fp,1); rec=tpr
    # AP = Σ ΔR·P (over bins from high to low, recall increasing)
    rec_full=np.concatenate([[0],rec[::-1]]); prec_full=np.concatenate([[1],prec[::-1]])
    ap=np.sum(np.diff(rec_full)*prec_full[1:])
    auroc=np.trapezoid(tpr[::-1],fpr[::-1])
    idx=np.searchsorted(tpr[::-1],0.95); fpr_asc=fpr[::-1]
    fpr95=float(fpr_asc[min(idx,len(fpr_asc)-1)])
    return dict(AP=float(ap),AUROC=float(auroc),FPR95=fpr95)

import json as _json
def _read_exact(R,pfx):
    """Read (AP,AUROC,FPR95) from the result json as a point-estimate sanity comparison."""
    e={}
    def g(path,*ks):
        try:
            d=_json.load(open(f"{R}/{path}"))
            for k in ks: d=d[k]
            return (float(d["AP"]),float(d["AUROC"]),float(d["FPR95"]))
        except Exception: return None
    e["DINOv2kNN"]=g(f"dinov2_knn_{pfx}test.json","pixel")
    for m in ["MSP","MaxLogit","Energy","SML"]: e[m]=g(f"baselines_seghead_{pfx}test.json",m)
    e["cDNP_k5"]=g(f"cdnp_{pfx}test.json","k5")
    e["Mahalanobis"]=g(f"mahalanobis_{pfx}test.json","result") or g(f"mahalanobis_{pfx}test.json","test")
    return {k:v for k,v in e.items() if v is not None}

def main(dataset="goose"):
    from datasets import get_dataset
    R=f"{ROOT}/results"; pfx="" if dataset=="goose" else f"{dataset}_"
    _,onto_loader=get_dataset(dataset); o=onto_loader(); test_lut=np.full(256,IGNORE,np.uint8)
    for k,v in o["test_lut"].items(): test_lut[k]=v
    dn=np.load(f"{R}/dinov2_scores_{pfx}test.npz"); keys=dn["keys"]
    gt_all=test_lut[keys]                                  # (F,H,W) 0/1/255
    F=keys.shape[0]
    EXACT=_read_exact(R,pfx)
    sources={"DINOv2kNN":(dn,"scores")}
    sg=np.load(f"{R}/seghead_scores_{pfx}test.npz")
    for m in ["MSP","MaxLogit","Energy","SML"]: sources[m]=(sg,m)
    for mname,fn in [("cDNP_k5",f"cdnp_scores_{pfx}test.npz"),("Mahalanobis",f"mahalanobis_scores_{pfx}test.npz")]:
        try: sources[mname]=(np.load(f"{R}/{fn}"),"scores")
        except FileNotFoundError: pass

    rng=np.random.default_rng(0); out={}
    print(f"{'method':12s} {'AP[95%CI]':>22s} {'AUROC[95%CI]':>22s} {'FPR95[95%CI]':>22s}  sanity")
    for mname,(npz,key) in sources.items():
        S=npz[key].astype(np.float32)
        # quantile edges: each bin equal mass, both dense region and top region have high resolution -> accurate AP (non-uniform edges OK, metrics only use counts)
        samp=S[gt_all!=255][::10]
        edges=np.unique(np.quantile(samp.astype(np.float64),np.linspace(0,1,NBINS+1)))
        edges[0]-=1e-6; edges[-1]+=1e-6                          # include boundaries
        nb=len(edges)-1                                          # actual number of bins (variable after quantile deduplication)
        # two histograms per frame
        idh=np.zeros((F,nb)); oodh=np.zeros((F,nb))
        for i in range(F):
            s=S[i]; g=gt_all[i]
            idh[i]=np.histogram(s[g==0],bins=edges)[0]
            oodh[i]=np.histogram(s[g==1],bins=edges)[0]
        # point estimate (self-check)
        pe=metrics_from_hist(idh.sum(0),oodh.sum(0),edges)
        ex=EXACT.get(mname)
        ok=("✅" if (abs(pe['AP']-ex[0])<0.01 and abs(pe['AUROC']-ex[1])<0.01 and abs(pe['FPR95']-ex[2])<0.02) else "🔴mismatch") if ex else "—"
        # bootstrap
        boot={"AP":[],"AUROC":[],"FPR95":[]}
        for _ in range(NBOOT):
            idx=rng.integers(0,F,F)
            r=metrics_from_hist(idh[idx].sum(0),oodh[idx].sum(0),edges)
            for kk in boot: boot[kk].append(r[kk])
        def ci(v): v=np.array(v); return (float(np.mean(v)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5)))
        cis={kk:ci(boot[kk]) for kk in boot}
        out[mname]=dict(point=pe,exact=(dict(AP=ex[0],AUROC=ex[1],FPR95=ex[2]) if ex else None),ci=cis)
        f=lambda t:f"{t[0]:.3f}[{t[1]:.3f},{t[2]:.3f}]"
        print(f"{mname:12s} {f(cis['AP']):>22s} {f(cis['AUROC']):>22s} {f(cis['FPR95']):>22s}  {ok}")
        del S,idh,oodh
    fn="main_table_ci.json" if dataset=="goose" else f"main_table_ci_{dataset}.json"
    json.dump(out,open(f"{ROOT}/results/{fn}","w"),indent=2)
    print(f"[saved] results/{fn}")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp"]); a=ap.parse_args()
    main(a.dataset)
