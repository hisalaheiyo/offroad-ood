#!/usr/bin/env python
"""
Outlier Exposure training (targets the root cause of head-confusion): frozen DINOv2 + linear head, pasting COCO objects onto off-road images during training,
outlier pixels use an OE loss (softmax -> uniform = maximum entropy) → the head learns to be uncertain about novel objects → anomaly scores (MSP/MaxLogit/Energy) rise.
ID pixels use normal CE. Expected improvement: WS logit collapse + vegetation confusion (the head no longer confidently misclassifies anomalies).
Usage (GPU): python train_seghead_oe.py --dataset goose --lam_oe 0.5 --p_oe 0.5
"""
import sys, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ontology import ROOT, IGNORE
from dinov2_baseline import load_model, DEV
from train_seghead import raw_patch_feats, target_to_patch
from coco_outliers import CocoOutliers, paste_outlier

def oe_head_path(ds, lam=0.5):
    tag=f"_l{lam}"; return f"{ROOT}/results/seghead_oe{tag}.pt" if ds=="goose" else f"{ROOT}/results/seghead_oe{tag}_{ds}.pt"
COCO_IMG=f"{ROOT}/data/coco/val2017"; COCO_ANN=f"{ROOT}/data/coco/annotations/instances_val2017.json"

def outlier_patch_mask(outlier_hw, gh, gw, thr=0.5):
    """Downsample the pixel-level outlier mask to the patch grid: a patch is an outlier patch if its outlier share > thr."""
    t=torch.from_numpy(outlier_hw.astype(np.float32))[None,None]
    frac=F.adaptive_avg_pool2d(t,(gh,gw))[0,0]
    return (frac>thr).to(DEV)

def main(dataset="goose", epochs=3, lr=1e-3, lam_oe=0.5, p_oe=0.5, seed=0):
    from datasets import get_dataset
    torch.manual_seed(seed); np.random.seed(seed)
    DS,onto_loader=get_dataset(dataset); onto=onto_loader(); NC=onto["num_train"]
    m,mean,std=load_model(); head=nn.Linear(384,NC).to(DEV)
    opt=torch.optim.AdamW(head.parameters(),lr=lr)
    co=CocoOutliers(COCO_IMG, COCO_ANN, seed=seed)
    rng=np.random.default_rng(seed)
    ds=DS("train",mode="train"); N=len(ds); t0=time.time()
    for ep in range(epochs):
        order=np.random.permutation(N); tot_ce=tot_oe=0; nb=0
        for j,i in enumerate(order):
            s=ds[int(i)]; img=s["image"]; lab=s["target"].copy()    # pixel ID train_id(255 ignore/OOD-masked)
            outlier=None
            if rng.random()<p_oe:
                crop,cm=co.sample(); img,outlier=paste_outlier(img,crop,cm,rng)
                lab[outlier]=255                                     # outlier pixels do not participate in ID CE
            f,(gh,gw)=raw_patch_feats(m,mean,std,img)
            logits=head(f.reshape(-1,384))                          # (gh*gw, C)
            tgt=target_to_patch(lab,gh,gw,nc=NC)                    # ID classes (255=ignore)
            ce=F.cross_entropy(logits, tgt.reshape(-1), ignore_index=255)
            loss=ce; oe_val=0.0
            if outlier is not None:
                opm=outlier_patch_mask(outlier,gh,gw).reshape(-1)
                if opm.any():
                    logp=F.log_softmax(logits[opm],dim=1)           # (n_out,C)
                    oe=-logp.mean()                                 # CE to a uniform distribution = maximum entropy
                    loss=ce+lam_oe*oe; oe_val=oe.item()
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); opt.step()
                tot_ce+=ce.item(); tot_oe+=oe_val; nb+=1
            if j%500==0: print(f"  ep{ep} {j}/{N} ce={tot_ce/max(nb,1):.3f} oe={tot_oe/max(nb,1):.3f} ({time.time()-t0:.0f}s)",flush=True)
        print(f"epoch{ep} ce={tot_ce/max(nb,1):.4f} oe={tot_oe/max(nb,1):.4f}",flush=True)
    hp=oe_head_path(dataset,lam_oe); torch.save(head.state_dict(),hp)
    print(f"[saved] {hp} (lam_oe={lam_oe} p_oe={p_oe})")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp"])
    ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--lam_oe",type=float,default=0.5)
    ap.add_argument("--p_oe",type=float,default=0.5); ap.add_argument("--seed",type=int,default=0); a=ap.parse_args()
    main(a.dataset,a.epochs,lam_oe=a.lam_oe,p_oe=a.p_oe,seed=a.seed)
