#!/usr/bin/env python
"""
Mahalanobis OOD baseline (Lee et al. NeurIPS 2018), frozen DINOv2 raw patch features, 20 ID classes.
fit: ID train patch -> per-class mean μ_c + tied (shared) covariance Σ (float64 sufficient statistics to prevent numerical cancellation).
eval: anomaly score = min_c (f-μ_c)^T Σ^-1 (f-μ_c)  (squared Mahalanobis distance to the nearest ID class, larger = more OOD).
self-check: Σ invertible with no NaN; on val the OOD-score median > ID-score median; AUROC.
Usage (GPU): python mahalanobis.py
Correctness essentials: raw features (not normalized); float64 statistics; Σ regularization (+εI).
"""
import json, time, numpy as np, torch, torch.nn.functional as F
from ontology import IGNORE, ROOT
from datasets import get_dataset
from dinov2_baseline import load_model, DEV
from train_seghead import raw_patch_feats, target_to_patch
import metrics as M

D=384

@torch.no_grad()
def fit(m,mean,std,onto,DS):
    """one-pass sufficient statistics -> μ_c(C×D), tied Σ(D×D), Σ^-1. float64 to prevent cancellation."""
    C=onto["num_train"]
    sumf=torch.zeros(C,D,dtype=torch.float64,device=DEV)      # per-class Σf
    cnt =torch.zeros(C,dtype=torch.float64,device=DEV)
    sumff=torch.zeros(D,D,dtype=torch.float64,device=DEV)     # global Σ f f^T
    ds=DS("train",mode="train"); t0=time.time()
    for i in range(len(ds)):
        s=ds[i]; f,(gh,gw)=raw_patch_feats(m,mean,std,s["image"])   # (gh,gw,D)
        tgt=target_to_patch(s["target"],gh,gw,nc=C).reshape(-1)     # (gh*gw,) 0..C-1 or 255
        X=f.reshape(-1,D).double()
        idm=tgt<C                                                   # ID patches
        if idm.any():
            Xi=X[idm]; ti=tgt[idm]
            sumf.index_add_(0,ti,Xi); cnt.index_add_(0,ti,torch.ones(Xi.shape[0],dtype=torch.float64,device=DEV))
            sumff+=Xi.T@Xi
        if i%500==0: print(f"  fit {i}/{len(ds)} ({time.time()-t0:.0f}s)",flush=True)
    N=cnt.sum(); mu=sumf/cnt.clamp(min=1)[:,None]                   # (C,D)
    # tied Σ = (Σ_total ffT - Σ_c n_c μ_c μ_c^T)/N
    Sclass=torch.zeros(D,D,dtype=torch.float64,device=DEV)
    for c in range(C): Sclass+=cnt[c]*torch.outer(mu[c],mu[c])
    Sigma=(sumff - Sclass)/N
    eps=1e-6*torch.diag(Sigma).mean()
    Sigma_reg=Sigma + eps*torch.eye(D,dtype=torch.float64,device=DEV)
    Sinv=torch.linalg.inv(Sigma_reg)
    # self-check: invertibility
    chk=(Sigma_reg@Sinv); err=(chk-torch.eye(D,device=DEV,dtype=torch.float64)).abs().max()
    print(f"[fit] N={int(N)} C={C} Σ diag mean {torch.diag(Sigma).mean():.4f} inverse error {err:.2e} {'✅' if err<1e-3 else '🔴'} ({time.time()-t0:.0f}s)",flush=True)
    return mu.float(), Sinv.float()

@torch.no_grad()
def maha_score(m,mean,std,mu,Sinv,img):
    f,(gh,gw)=raw_patch_feats(m,mean,std,img); X=f.reshape(-1,D)   # (P,D)
    # d_c = (X-μ_c) Sinv (X-μ_c)^T for each class, take min
    P=X.shape[0]; mind=torch.full((P,),1e30,device=DEV)
    for c in range(mu.shape[0]):
        R=X-mu[c]                                                  # (P,D)
        d=((R@Sinv)*R).sum(1)                                      # (P,)
        mind=torch.minimum(mind,d)
    sc=mind.reshape(gh,gw)
    return F.interpolate(sc[None,None],size=img.shape[:2],mode="bilinear",align_corners=False)[0,0].cpu().numpy()

def main(split="test", dataset="goose"):
    DS,onto_loader=get_dataset(dataset); onto=onto_loader()
    test_lut=np.full(256,IGNORE,np.uint8)
    for k,v in onto["test_lut"].items(): test_lut[k]=v
    m,mean,std=load_model()
    mu,Sinv=fit(m,mean,std,onto,DS)
    # self-check: on val, OOD-score median > ID-score median
    dsv=DS("val",mode="test"); vs=[];vg=[]
    for i in range(0,len(dsv),3):
        s=dsv[i]; sc=maha_score(m,mean,std,mu,Sinv,s["image"])[::4,::4]; gt=s["target"][::4,::4]
        vs.append(sc.ravel());vg.append(gt.ravel())
    vs=np.concatenate(vs);vg=np.concatenate(vg)
    print(f"[self-check] val OOD median {np.median(vs[vg==1]):.2f} vs ID median {np.median(vs[vg==0]):.2f} (OOD should be > ID) | magnitude max {vs.max():.1f}",flush=True)
    # eval target split (fit always on train)
    dst=DS(split,mode="test"); N=len(dst); sc_l=[];gts=[];keys=[];t0=time.time()
    for i in range(N):
        s=dst[i]; sc=maha_score(m,mean,std,mu,Sinv,s["image"])[::4,::4].astype(np.float16)
        sc_l.append(sc); gts.append(s["target"][::4,::4]); keys.append(s["label_key"][::4,::4].copy())
        if i%300==0: print(f"  {split} {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
    r=M.pixel_pr_auroc_fpr95(sc_l,gts)
    print(f"\n=== Mahalanobis ({split}) ===\n  AP={r['AP']:.4f} AUROC={r['AUROC']:.4f} FPR95={r['FPR95']:.4f}")
    pfx="" if dataset=="goose" else f"{dataset}_"
    json.dump(dict(method="mahalanobis",dataset=dataset,split=split,result=r),open(f"{ROOT}/results/mahalanobis_{pfx}{split}.json","w"),indent=2)
    np.savez_compressed(f"{ROOT}/results/mahalanobis_scores_{pfx}{split}.npz",scores=np.stack(sc_l),keys=np.stack(keys))
    print(f"[saved] mahalanobis_{pfx}{split}.json + mahalanobis_scores_{pfx}{split}.npz")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--split",default="test")
    ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp","goose_v2"]); a=ap.parse_args()
    main(a.split, dataset=a.dataset)
