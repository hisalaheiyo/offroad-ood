#!/usr/bin/env python
"""
Leakage ablation (multi-dataset, controlled): shows that "same-scene background entering the bank inflates performance" -> only the leakage-free split is realistic (core benchmark argument).
Controlled: the same batch of eval frames (half-A of test), varying only whether the bank contains same-scene background:
  CLEAN bank = ID-patches from train frames only
  LEAKY bank = 50% train + 50% half-B of test (different frames from half-A but same scenes = simulates leakage)
testA/testB alternate by global index (even/odd within each scene) -> A, B share all scenes. Both banks have the same size 1M (fair).
Usage (GPU): python leakage_ablation_multi.py --dataset rellis
"""
import time, argparse, json, numpy as np, torch
from datasets import get_dataset
from dinov2_baseline import load_model, extract_patches, patch_purity_mask, score_frame, DEV
from ontology import IGNORE, ROOT
import metrics as M

@torch.no_grad()
def sample_patches(m,mean,std,onto,ds,indices,target,per_frame,seed=0):
    rng=np.random.default_rng(seed); feats=[]; ntot=0
    for i in indices:
        s=ds[i]; f,(gh,gw)=extract_patches(m,mean,std,s["image"])
        pure=patch_purity_mask(s["label_key"],gh,gw,onto); idx=np.nonzero(pure)[0]
        if len(idx)>per_frame: idx=rng.choice(idx,per_frame,replace=False)
        if len(idx): feats.append(f[idx].cpu()); ntot+=len(idx)
        if ntot>=target: break
    return torch.cat(feats)[:target].contiguous()

@torch.no_grad()
def eval_on(m,mean,std,bank,test_lut,ds,indices,downsample=4):
    scores=[];gts=[]
    for i in indices:
        s=ds[i]; sc=score_frame(m,mean,std,bank,s["image"])[::downsample,::downsample].astype(np.float16)
        gt=test_lut[s["label_key"]][::downsample,::downsample]; scores.append(sc); gts.append(gt)
    return M.pixel_pr_auroc_fpr95(scores,gts)

def main(dataset="goose"):
    t0=time.time()
    DS,ol=get_dataset(dataset); onto=ol()
    test_lut=np.full(256,IGNORE,np.uint8)
    for k,v in onto["test_lut"].items(): test_lut[k]=v
    m,mean,std=load_model()
    train_ds=DS("train",mode="test"); test_ds=DS("test",mode="test")
    Ntr=len(train_ds); Nte=len(test_ds)
    A=[i for i in range(Nte) if i%2==0]; B=[i for i in range(Nte) if i%2==1]   # alternate -> same scenes shared
    print(f"[{dataset}] train{Ntr} test{Nte} → testA{len(A)} testB{len(B)}",flush=True)
    # CLEAN=1M train; LEAKY=0.5M train + 0.5M testB
    clean=sample_patches(m,mean,std,onto,train_ds,list(range(Ntr)),1_000_000,400).to(DEV)
    print(f"clean bank{tuple(clean.shape)} ({time.time()-t0:.0f}s)",flush=True)
    tb_tr=sample_patches(m,mean,std,onto,train_ds,list(range(Ntr)),500_000,400,seed=1)
    tb_te=sample_patches(m,mean,std,onto,test_ds,B,500_000,800,seed=2)
    leaky=torch.cat([tb_tr,tb_te])[:1_000_000].contiguous().to(DEV)
    print(f"leaky bank{tuple(leaky.shape)} (same-scene{tb_te.shape[0]/leaky.shape[0]*100:.0f}%) ({time.time()-t0:.0f}s)",flush=True)
    rc=eval_on(m,mean,std,clean,test_lut,test_ds,A); print(f"CLEAN(leakage-free): AP{rc['AP']:.4f} AUROC{rc['AUROC']:.4f} FPR95{rc['FPR95']:.4f}",flush=True)
    rl=eval_on(m,mean,std,leaky,test_lut,test_ds,A); print(f"LEAKY(with same-scene): AP{rl['AP']:.4f} AUROC{rl['AUROC']:.4f} FPR95{rl['FPR95']:.4f}",flush=True)
    print(f"  ΔAP={rl['AP']-rc['AP']:+.3f} ΔFPR95={rl['FPR95']-rc['FPR95']:+.3f} → leakage inflates performance (only the leakage-free split is realistic)")
    pfx="" if dataset=="goose" else f"{dataset}_"
    json.dump(dict(clean=rc,leaky=rl,nA=len(A),nB=len(B)),open(f"{ROOT}/results/leakage_ablation_{pfx}.json","w"),indent=2)
    print(f"[saved] leakage_ablation_{pfx}.json ({time.time()-t0:.0f}s)")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd"]); a=ap.parse_args()
    main(a.dataset)
