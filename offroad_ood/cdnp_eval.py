#!/usr/bin/env python
"""
cDNP baseline (Galesso ICCVW23): anomaly = mean top-k nearest-neighbor distance from each patch to the ID bank on frozen DINOv2 features.
score = 1 - mean(top-k cosine sim). k=1 is exactly equal to the existing DINOv2 kNN (sanity).
Run the k=1/5/15 comparison, dump k=5 score+keys for failure analysis.
Usage (GPU): python cdnp_eval.py
"""
import json, time, gc, numpy as np, torch, torch.nn.functional as F
from ontology import IGNORE, ROOT
from dinov2_baseline import load_model, extract_patches, bank_path, DEV
from datasets import get_dataset
import metrics as M

@torch.no_grad()
def score_knn(m,mean,std,bank,img,ks,B=2048):
    """Returns {k: score_map(H,W)}. Single forward pass, multiple k reuse the same topk."""
    f,(gh,gw)=extract_patches(m,mean,std,img); bT=bank.T; kmax=max(ks)
    topk=torch.empty(f.shape[0],kmax,device=DEV)
    for s in range(0,f.shape[0],B):
        sim=f[s:s+B]@bT                       # (b,|bank|)
        topk[s:s+B]=sim.topk(kmax,dim=1).values
        del sim
    out={}
    for k in ks:
        sc=(1-topk[:,:k].mean(1)).reshape(gh,gw)
        out[k]=F.interpolate(sc[None,None],size=img.shape[:2],mode="bilinear",align_corners=False)[0,0].cpu().numpy()
    return out

def main(ks=(1,5,15), dump_k=5, split="test", dataset="goose"):
    DS,onto_loader=get_dataset(dataset); onto=onto_loader()
    test_lut=np.full(256,IGNORE,np.uint8)
    for kk,v in onto["test_lut"].items(): test_lut[kk]=v
    m,mean,std=load_model(); bank=torch.load(bank_path(dataset)).to(DEV)
    ds=DS(split,mode="test"); N=len(ds)
    buf={k:[] for k in ks}; gts=[]; keys=[]; t0=time.time()
    for i in range(N):
        s=ds[i]; maps=score_knn(m,mean,std,bank,s["image"],ks)
        for k in ks: buf[k].append(maps[k][::4,::4].astype(np.float16))
        gts.append(s["target"][::4,::4]); keys.append(s["label_key"][::4,::4].copy())
        if i%300==0: print(f"  {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
    print("\n=== cDNP (test) ===")
    res={}; dump_scores=None
    for k in ks:
        r=M.pixel_pr_auroc_fpr95(buf[k],gts); res[f"k{k}"]=r
        tag=" (= existing kNN sanity)" if k==1 else ""
        print(f"  k={k:2d}: AP={r['AP']:.4f} AUROC={r['AUROC']:.4f} FPR95={r['FPR95']:.4f}{tag}",flush=True)
        if k==dump_k: dump_scores=np.stack(buf[k])      # keep the one to dump
        buf[k]=None; gc.collect()                       # release immediately after computation (avoids 3 sets of 400M pixels co-resident -> 100G OOM)
    print(f"  existing DINOv2kNN: AP=0.3604 AUROC=0.8721 FPR95=0.6778 <- k=1 should be approximately this")
    pfx="" if dataset=="goose" else f"{dataset}_"
    json.dump(res,open(f"{ROOT}/results/cdnp_{pfx}{split}.json","w"),indent=2)
    np.savez_compressed(f"{ROOT}/results/cdnp_scores_{pfx}{split}.npz", scores=dump_scores, keys=np.stack(keys), k=dump_k)
    print(f"[saved] cdnp_{pfx}{split}.json + cdnp_scores_{pfx}{split}.npz(k={dump_k})")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--split",default="test")
    ap.add_argument("--ks",default="1,5,15",help="comma-separated; if WS memory is tight, use '5' to compute only the dumped k")
    ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp","goose_v2"]); a=ap.parse_args()
    ks=tuple(int(x) for x in a.ks.split(",")); assert 5 in ks, "dump_k=5 must be in ks"
    main(ks=ks, split=a.split, dataset=a.dataset)
