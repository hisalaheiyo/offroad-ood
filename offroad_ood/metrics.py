#!/usr/bin/env python
"""
OffRoad-OOD — evaluation suite core (method-agnostic: inputs anomaly score + GT{0,1,255})
Metrics:
  pixel-level: AP(AuPRC), AUROC, FPR@95TPR        (ignores 255)
  component-level: SMIYC-style sIoU/PPV/F1, averaged over τ∈{0.25..0.75}
  bootstrap 95% CI (frame-level resampling)
Distance stratification: out of scope for this release (would require LiDAR projection).
All functions aggregate over single or multiple frames; multi-frame uses frame-level score/GT concatenation (pixel metrics) or per-frame component accumulation.
"""
import numpy as np
from scipy import ndimage

# ---------- pixel-level ----------
def _valid(score, gt):
    m = gt != 255
    return score[m].ravel().astype(np.float32), gt[m].ravel().astype(np.int8)

def pixel_pr_auroc_fpr95(scores, gts):
    """scores,gts: equal-length list of 2D arrays (or a single one). Returns AP, AUROC, FPR95."""
    if not isinstance(scores, list): scores=[scores]; gts=[gts]
    s_all=[]; y_all=[]
    for s,g in zip(scores,gts):
        s,y=_valid(s,g); s_all.append(s); y_all.append(y)
    s=np.concatenate(s_all); y=np.concatenate(y_all)
    if y.sum()==0 or y.sum()==len(y): return dict(AP=float('nan'),AUROC=float('nan'),FPR95=float('nan'),n_pos=int(y.sum()))
    order=np.argsort(-s); y=y[order]; s=s[order]
    P=y.sum(); N=len(y)-P
    tp=np.cumsum(y); fp=np.cumsum(1-y)
    tpr=tp/P; fpr=fp/N
    prec=tp/np.maximum(tp+fp,1)
    rec=tpr
    # AP = ∑(R_i - R_{i-1})*P_i
    ap=np.sum(np.diff(np.concatenate([[0],rec]))*prec)
    # AUROC trapezoidal
    auroc=np.trapezoid(tpr, fpr)
    # FPR@95TPR: minimum fpr s.t. tpr>=0.95
    idx=np.searchsorted(tpr,0.95)
    fpr95=float(fpr[min(idx,len(fpr)-1)])
    return dict(AP=float(ap),AUROC=float(auroc),FPR95=fpr95,n_pos=int(P))

# ---------- component-level (SMIYC-style) ----------
def _components(mask):
    lab,n=ndimage.label(mask)
    return lab,n

def component_f1_single(score, gt, thr, taus=None, min_pix=10):
    """Single frame: at binary threshold thr, match GT connected components vs predicted connected components via sIoU/PPV. Returns per-τ (TP_gt, FN, FP, TP_pred approx)."""
    if taus is None: taus=np.arange(0.25,0.751,0.05)
    valid=gt!=255
    pred=(score>=thr)&valid
    gtm=(gt==1)&valid
    gl,gn=_components(gtm); pl,pn=_components(pred)
    res={float(t):dict(tp=0,fn=0,fp=0) for t in taus}
    # predicted component areas & intersection with each gt
    g_sizes=ndimage.sum(np.ones_like(gl),gl,index=np.arange(1,gn+1)) if gn else np.array([])
    p_sizes=ndimage.sum(np.ones_like(pl),pl,index=np.arange(1,pn+1)) if pn else np.array([])
    # filter out tiny components
    g_valid=[i+1 for i in range(gn) if g_sizes[i]>=min_pix]
    p_valid=[i+1 for i in range(pn) if p_sizes[i]>=min_pix]
    # GT-side sIoU (correct SMIYC approach: use only the union of predicted components intersecting this GT; FPs elsewhere do not affect this GT)
    pvalid_mask=np.isin(pl,p_valid) if p_valid else np.zeros_like(pl,bool)
    for gi in g_valid:
        gmask=gl==gi
        hit=np.unique(pl[gmask & pvalid_mask])      # (valid) predicted component labels intersecting this GT
        hit=hit[hit>0]
        if len(hit)==0:                              # no predicted coverage -> sIoU=0 -> all FN
            for t in taus: res[float(t)]["fn"]+=1
            continue
        Phat=np.isin(pl,hit)                         # union of these predicted components
        inter=(gmask&Phat).sum(); union=(gmask|Phat).sum()
        siou=inter/max(union,1)
        for t in taus:
            if siou>float(t): res[float(t)]["tp"]+=1
            else: res[float(t)]["fn"]+=1
    # prediction side: PPV=|p∩gt_union|/|p|, counted as FP if below τ (local to this predicted component, unaffected by elsewhere)
    gtU=gtm
    for pi in p_valid:
        pmask=pl==pi
        ppv=(pmask&gtU).sum()/max(pmask.sum(),1)
        for t in taus:
            if ppv<=float(t): res[float(t)]["fp"]+=1
    return res

def component_f1_aggregate(scores, gts, thr, taus=None, min_pix=10):
    if taus is None: taus=np.arange(0.25,0.751,0.05)
    if not isinstance(scores,list): scores=[scores]; gts=[gts]
    acc={float(t):dict(tp=0,fn=0,fp=0) for t in taus}
    for s,g in zip(scores,gts):
        r=component_f1_single(s,g,thr,taus,min_pix)
        for t in taus:
            for k in("tp","fn","fp"): acc[float(t)][k]+=r[float(t)][k]
    f1s=[]
    for t in taus:
        a=acc[float(t)]; tp,fn,fp=a["tp"],a["fn"],a["fp"]
        f1=tp/max(tp+0.5*(fn+fp),1e-9); f1s.append(f1)
    return dict(component_F1=float(np.mean(f1s)), per_tau=dict(zip([float(t) for t in taus],f1s)))

# ---------- bootstrap CI (frame-level) ----------
def bootstrap_ci(scores, gts, metric_fn, n_boot=1000, seed=0, **kw):
    rng=np.random.default_rng(seed); N=len(scores); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,N,N)
        v=metric_fn([scores[i] for i in idx],[gts[i] for i in idx],**kw)
        vals.append(v)
    vals=np.array([x for x in vals if not np.isnan(x)])
    return dict(mean=float(vals.mean()), lo=float(np.percentile(vals,2.5)), hi=float(np.percentile(vals,97.5)))

# distance-stratified evaluation is out of scope here (would require projecting LiDAR onto the image
# to estimate the distance of each OOD instance); not used by any result in the paper.
def distance_stratified(*a,**k): raise NotImplementedError("distance stratification requires LiDAR projection; out of scope for this release")

# ---------- self-check: validate metric correctness on synthetic data ----------
if __name__=="__main__":
    rng=np.random.default_rng(0)
    # construct: perfect scores -> AP≈1, FPR95≈0; random scores -> AP≈positive rate, AUROC≈0.5
    H,W=64,64
    gts=[]; perf=[]; rand=[]
    for _ in range(20):
        g=np.zeros((H,W),np.int16)
        g[20:30,20:30]=1   # one square anomaly
        g[0:5,0:5]=255     # a bit of ignore
        s=g.astype(float).copy(); s[g==1]=10; s[g==0]=-10  # perfectly separable
        gts.append(g); perf.append(s+rng.normal(0,0.01,(H,W)))
        rand.append(rng.normal(0,1,(H,W)))
    print("=== perfect scores (expect AP≈1, AUROC≈1, FPR95≈0) ===")
    print({k:round(v,4) for k,v in pixel_pr_auroc_fpr95(perf,gts).items()})
    print("=== random scores (expect AUROC≈0.5) ===")
    print({k:round(v,4) for k,v in pixel_pr_auroc_fpr95(rand,gts).items()})
    print("=== component-F1 perfect (thr 0, expect ≈1) ===")
    print({k:(round(v,4) if isinstance(v,float) else v) for k,v in component_f1_aggregate(perf,gts,thr=0.0).items() if k=='component_F1'})
    print("=== bootstrap CI on AP (perfect) ===")
    print({k:round(v,4) for k,v in bootstrap_ci(perf,gts,lambda s,g:pixel_pr_auroc_fpr95(s,g)['AP']).items()})
