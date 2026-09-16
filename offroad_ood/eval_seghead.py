#!/usr/bin/env python
"""
Derive classic urban OOD baselines from the logits of a frozen DINOv2 + linear head, evaluated on OffRoad-OOD test:
  MSP      = 1 - max_softmax
  MaxLogit = -max_logit
  Energy   = -logsumexp(logits)
  SML      = -(maxlogit - mu_c)/sigma_c   (c=argmax; mu/sigma from val ID pixels, per predicted class)
Usage (GPU): python eval_seghead.py
"""
import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ontology import IGNORE, ROOT
from dinov2_baseline import load_model, DEV
from train_seghead import raw_patch_feats, head_path
from datasets import get_dataset
import metrics as M

@torch.no_grad()
def frame_logits(m,mean,std,head,img):
    f,(gh,gw)=raw_patch_feats(m,mean,std,img)
    lo=head(f.reshape(-1,384)).reshape(gh,gw,-1).permute(2,0,1)   # (C,gh,gw)
    up=F.interpolate(lo[None],size=img.shape[:2],mode="bilinear",align_corners=False)[0]
    return up   # (C,H,W)

def scores_from_logits(lo):
    # lo: (C,H,W) tensor
    msp = 1 - F.softmax(lo,0).max(0).values
    maxl= -lo.max(0).values
    en  = -torch.logsumexp(lo,0)
    return dict(MSP=msp, MaxLogit=maxl, Energy=en)

def sml_stats(m,mean,std,head,DS,NC,t0=None):
    """val ID pixels -> per-predicted-class maxlogit mu/sigma (SML standardization statistics). float64 to avoid numerical cancellation. Returns (mu,sig)."""
    import time as _t; t0=t0 if t0 is not None else _t.time()
    dsv=DS("val",mode="test")
    sums=torch.zeros(NC,device=DEV,dtype=torch.float64);sqs=torch.zeros(NC,device=DEV,dtype=torch.float64);cnt=torch.zeros(NC,device=DEV,dtype=torch.float64)
    for i in range(len(dsv)):
        s=dsv[i]; lo=frame_logits(m,mean,std,head,s["image"]); gt=torch.from_numpy(s["target"]).to(DEV)
        ml,pred=lo.max(0); idm=(gt==0)            # ID pixels
        ml=ml[idm].double();pc=pred[idm]
        sums.index_add_(0,pc,ml); sqs.index_add_(0,pc,ml*ml); cnt.index_add_(0,pc,torch.ones_like(ml))
        if i%400==0: print(f"  sml-cal {i}/{len(dsv)} ({_t.time()-t0:.0f}s)",flush=True)
    mu=sums/cnt.clamp(min=1); var=(sqs/cnt.clamp(min=1)-mu*mu).clamp(min=0); sig=var.sqrt()
    # robust: under-sampled classes (< 100 pixels summed over frames) use the global statistics; set sigma to a floor relative to the global to avoid blow-up
    gmu=sums.sum()/cnt.sum().clamp(min=1); gvar=(sqs.sum()/cnt.sum().clamp(min=1)-gmu*gmu).clamp(min=1e-6); gsig=gvar.sqrt()
    under=cnt<100; mu[under]=gmu; sig[under]=gsig
    sig=torch.maximum(sig, 0.1*gsig)
    print(f"SML stats: gmu{gmu:.2f} gsig{gsig:.2f} sig range[{sig.min():.3f},{sig.max():.3f}] under-sampled classes{int(under.sum())}/{NC} ({_t.time()-t0:.0f}s)",flush=True)
    return mu,sig

def main(split="test", dump=True, dataset="goose", coarse=None, seed=0, oe=False, oe_lam=0.5):
    DS,onto_loader=get_dataset(dataset); onto=onto_loader(); NC=onto["num_train"]
    if coarse:
        assert dataset in ("goose","rellis"),"granularity ablation supports goose/rellis"
        from coarse_groups import coarse_lut
        _,NC,gn=coarse_lut(onto,coarse,dataset); print(f"[coarse {coarse}/{dataset}] NC={NC}",flush=True)
    m,mean,std=load_model()
    if oe:
        from train_seghead_oe import oe_head_path; hp=oe_head_path(dataset,oe_lam); print(f"[OE head] {hp}",flush=True)
    else: hp=head_path(dataset,coarse,seed)
    head=nn.Linear(384,NC).to(DEV); head.load_state_dict(torch.load(hp)); head.eval()
    t0=time.time()
    # pass1: SML standardization statistics (from val ID pixels)
    mu,sig=sml_stats(m,mean,std,head,DS,NC,t0)
    # pass2: evaluate the target split (SML statistics always come from val)
    dst=DS(split,mode="test"); N=len(dst)
    methods=["MSP","MaxLogit","Energy","SML"]
    buf={k:[] for k in methods}; gts=[]; keys=[]
    for i in range(N):
        s=dst[i]; lo=frame_logits(m,mean,std,head,s["image"])
        sc=scores_from_logits(lo)
        ml,pred=lo.max(0); sml = -(ml - mu[pred])/sig[pred]
        sc["SML"]=sml
        gt=s["target"][::4,::4]; gts.append(gt); keys.append(s["label_key"][::4,::4].copy())
        for k in methods: buf[k].append(sc[k].cpu().numpy()[::4,::4].astype(np.float16))
        if i%300==0: print(f"  {split} {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
    # SML self-check: magnitude should be O(1-10), OOD median should be > ID median
    sa=np.concatenate([x.astype(np.float32).ravel() for x in buf["SML"][:200]])
    ga=np.concatenate([g[::1,::1].ravel() for g in gts[:200]])
    print(f"\n[SML self-check] magnitude[{sa.min():.2f},{sa.max():.2f}] | OOD median{np.median(sa[ga==1]):.3f} vs ID median{np.median(sa[ga==0]):.3f} (OOD should be > ID)",flush=True)
    print("\n=== BASELINE TABLE (test) ===")
    print(f"{'method':10s} {'AP':>7s} {'AUROC':>7s} {'FPR95':>7s}")
    out={}
    for k in methods:
        r=M.pixel_pr_auroc_fpr95(buf[k],gts); out[k]=r
        print(f"{k:10s} {r['AP']:7.4f} {r['AUROC']:7.4f} {r['FPR95']:7.4f}")
    # append the existing DINOv2 kNN reference
    print(f"{'(DINOv2kNN)':10s} {0.3604:7.4f} {0.8721:7.4f} {0.6778:7.4f}  <- previous")
    pfx="" if dataset=="goose" else f"{dataset}_"
    tag=pfx+(f"coarse_{coarse}_" if coarse else "")+(f"seed{seed}_" if seed else "")+(f"oe_l{oe_lam}_" if oe else "")   # granularity/seed/OE-specific filename
    json.dump(out, open(f"{ROOT}/results/baselines_seghead_{tag}{split}.json","w"), indent=2)
    if dump:
        np.savez_compressed(f"{ROOT}/results/seghead_scores_{tag}{split}.npz",
                            keys=np.stack(keys), **{k:np.stack(buf[k]) for k in methods})
    print(f"[saved] baselines_seghead_{tag}{split}.json (+seghead_scores_{tag}{split}.npz)")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--split",default="test")
    ap.add_argument("--coarse",default=None,choices=["L11","L7","L4"],help="granularity ablation (goose only)")
    ap.add_argument("--seed",type=int,default=0,help="seed robustness")
    ap.add_argument("--oe",action="store_true",help="evaluate the OE-trained head")
    ap.add_argument("--oe_lam",type=float,default=0.5,help="lam of the OE head (selects the corresponding head)")
    ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp","goose_v2","goose_v3"]); a=ap.parse_args()
    main(a.split, dataset=a.dataset, coarse=a.coarse, seed=a.seed, oe=a.oe, oe_lam=a.oe_lam)
