#!/usr/bin/env python
"""
A-baseline unlock: frozen DINOv2 + linear segmentation head (20 ID classes, pixel-masking, leakage-free split)
After training, its logits derive the four classic urban OOD baselines MSP/MaxLogit/Energy/SML.
Patch-resolution training (target downsampled to the patch grid), saving compute.
Usage: python train_seghead.py [smoke|train] [--epochs 3]
"""
import sys, time, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ontology import load_goose_ontology, IGNORE, ROOT
from goose_dataset import GooseOOD
from dinov2_baseline import load_model, _prep, PATCH, DEV
HEAD_PATH=f"{ROOT}/results/seghead.pt"
def head_path(ds, coarse=None, seed=0):
    sd=f"_seed{seed}" if seed else ""                                # seed=0 = existing path (backward compatible)
    if coarse:
        pre="" if ds=="goose" else f"{ds}_"                          # avoid GOOSE/RELLIS coarsened-head conflict
        return f"{ROOT}/results/seghead_coarse_{pre}{coarse}{sd}.pt"
    base=HEAD_PATH[:-3] if ds=="goose" else f"{ROOT}/results/seghead_{ds}"
    return f"{base}{sd}.pt"

@torch.no_grad()
def raw_patch_feats(m, mean, std, img_np, proc_hw=(518,1036)):
    """Raw (un-normalized) patch features for the linear classification head. Returns (gh,gw,384)."""
    t,(gh,gw)=_prep(img_np, proc_hw); t=((t.to(DEV)-mean)/std)
    feat=m.forward_features(t)
    if feat.shape[1]==gh*gw+1: feat=feat[:,1:,:]
    return feat[0].reshape(gh,gw,-1), (gh,gw)

def target_to_patch(target_hw, gh, gw, nc=20):
    """Downsample pixel-level train_id (255=ignore) to (gh,gw): each patch takes the majority ID class (255 if all ignore). nc = number of ID classes."""
    t=torch.from_numpy(target_hw.astype(np.int64))[None,None].float()
    # Area-pool each train_id, take the class with the largest share and >0; ignore (255) handled separately
    H,W=target_hw.shape
    # one-hot over nc classes + ignore channel (nc=ignore)
    oh=torch.zeros(nc+1,H,W)  # 0..nc-1 classes, nc=ignore
    tt=torch.from_numpy(target_hw.astype(np.int64)); tt2=tt.clone(); tt2[tt2==255]=nc
    oh.scatter_(0, tt2[None], 1.0)
    pooled=F.adaptive_avg_pool2d(oh[None], (gh,gw))[0]   # (nc+1,gh,gw) per-class share
    cls=pooled[:nc].argmax(0)                            # majority ID class
    has_id=pooled[:nc].sum(0) > 0.0                      # does this patch have ID pixels?
    out=torch.where(has_id, cls, torch.full_like(cls,255))
    return out.to(DEV)                                   # (gh,gw) long

def main(mode="train", epochs=3, lr=1e-3, dataset="goose", coarse=None, seed=0):
    from datasets import get_dataset
    torch.manual_seed(seed); np.random.seed(seed)            # seed robustness: controls head initialization + training order
    DS,onto_loader=get_dataset(dataset); onto=onto_loader(); NC=onto["num_train"]
    clut=None
    if coarse:
        assert dataset in ("goose","rellis"), "granularity ablation supports goose/rellis"
        from coarse_groups import coarse_lut
        clut,NC,gn=coarse_lut(onto,coarse,dataset); print(f"[coarse {coarse}/{dataset}] NC={NC} groups={gn}",flush=True)
    hp=head_path(dataset, coarse, seed)
    m,mean,std=load_model()
    head=nn.Linear(384, NC).to(DEV)
    if mode=="smoke":
        ds=DS("train",mode="train")
        opt=torch.optim.AdamW(head.parameters(),lr=lr)
        for i in range(2):
            s=ds[i]; f,(gh,gw)=raw_patch_feats(m,mean,std,s["image"])
            tgt_raw=clut[s["target"]] if clut is not None else s["target"]
            tgt=target_to_patch(tgt_raw,gh,gw,nc=NC)
            logits=head(f.reshape(-1,384))
            loss=F.cross_entropy(logits, tgt.reshape(-1), ignore_index=255)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"smoke frame{i}: feat{tuple(f.shape)} tgt uniq{torch.unique(tgt).tolist()[:8]} loss{loss.item():.3f}")
        print("smoke ✅" if torch.isfinite(loss) else "smoke 🔴")
        return
    # train
    ds=DS("train",mode="train"); N=len(ds)
    opt=torch.optim.AdamW(head.parameters(),lr=lr)
    t0=time.time()
    for ep in range(epochs):
        order=np.random.permutation(N); tot=0; nb=0
        for j,i in enumerate(order):
            s=ds[int(i)]; f,(gh,gw)=raw_patch_feats(m,mean,std,s["image"])
            tgt_raw=clut[s["target"]] if clut is not None else s["target"]
            tgt=target_to_patch(tgt_raw,gh,gw,nc=NC)
            logits=head(f.reshape(-1,384))
            loss=F.cross_entropy(logits,tgt.reshape(-1),ignore_index=255)
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
            if j%500==0: print(f"  ep{ep} {j}/{N} loss{tot/max(nb,1):.3f} ({time.time()-t0:.0f}s)",flush=True)
        print(f"epoch{ep} mean_loss={tot/max(nb,1):.4f}",flush=True)
    torch.save(head.state_dict(), hp)
    print(f"[saved] {hp}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("mode",nargs="?",default="train",choices=["smoke","train"])
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--coarse",default=None,choices=["L11","L7","L4"],help="granularity ablation (goose only)")
    ap.add_argument("--seed",type=int,default=0,help="seed robustness")
    ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp","goose_v2","goose_v3"]); a=ap.parse_args()
    main(a.mode, a.epochs, dataset=a.dataset, coarse=a.coarse, seed=a.seed)
