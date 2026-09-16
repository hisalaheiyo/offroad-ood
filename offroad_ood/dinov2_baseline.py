#!/usr/bin/env python
"""
OffRoad-OOD — DINOv2 kNN baseline (training-free, in the PixOOD/cDNP family)
Idea: freeze DINOv2 ViT-S/14 patch features; build a memory bank from pure-ID patches of the train set;
      for each test patch, take the nearest-neighbor cosine distance to the bank -> anomaly score; upsample back to the original image and compare with GT.
Key: only use "pure-ID patches" (no OOD/ignore within that 14x14 block) to build the bank, to avoid contamination; downsample the bank to cap its size.
Usage:
  python dinov2_baseline.py build --max_bank 1000000 --per_frame 400   # build bank (GPU)
  python dinov2_baseline.py eval  --split test                          # evaluate (GPU)
  python dinov2_baseline.py smoke                                       # small CPU check (2 frames, verifies logic)
"""
import os, sys, json, time, argparse, numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from ontology import load_goose_ontology, IGNORE, ROOT
from goose_dataset import GooseOOD
from datasets import get_dataset

PATCH=14
MODEL="vit_small_patch14_dinov2.lvd142m"  # timm name; 384-dim
BANK_PATH=f"{ROOT}/results/dinov2_bank.pt"
def bank_path(ds): return BANK_PATH if ds=="goose" else f"{ROOT}/results/dinov2_bank_{ds}.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    import timm
    m=timm.create_model(MODEL, pretrained=True, num_classes=0, dynamic_img_size=True).eval().to(DEV)
    cfg=timm.data.resolve_model_data_config(m)
    mean=torch.tensor(cfg["mean"]).view(1,3,1,1).to(DEV)
    std =torch.tensor(cfg["std"]).view(1,3,1,1).to(DEV)
    return m, mean, std

def _prep(img_np, target_hw):
    """img HxWx3 uint8 -> 1x3xH'xW' (H',W' are multiples of 14), returns tensor + (gh,gw) patch-grid size.
    If the environment variable ASPECT_PROC=1: preserve the original aspect ratio (long side = max(target_hw)), to avoid a forced 2:1 distortion (small objects in WS/RELLIS)."""
    import os
    if os.environ.get("ASPECT_PROC")=="1":
        oh,ow=img_np.shape[:2]; L=max(target_hw)
        if ow>=oh: Wf=L; Hf=round(L*oh/ow)
        else: Hf=L; Wf=round(L*ow/oh)
        Hf=max(PATCH,(Hf//PATCH)*PATCH); Wf=max(PATCH,(Wf//PATCH)*PATCH)
    else:
        H,W = target_hw
        Hf=(H//PATCH)*PATCH; Wf=(W//PATCH)*PATCH
    t=torch.from_numpy(img_np.copy()).permute(2,0,1).float().unsqueeze(0)/255.
    t=F.interpolate(t, size=(Hf,Wf), mode="bilinear", align_corners=False)
    return t, (Hf//PATCH, Wf//PATCH)

@torch.no_grad()
def extract_patches(m, mean, std, img_np, proc_hw=(518,1036)):
    """Returns L2-normalized patch features (gh*gw, 384) and the grid (gh,gw)."""
    t,(gh,gw)=_prep(img_np, proc_hw); t=((t.to(DEV)-mean)/std)
    feat=m.forward_features(t)            # timm DINOv2: (1, 1+gh*gw, 384) includes cls
    if feat.shape[1]==gh*gw+1: feat=feat[:,1:,:]     # drop cls
    elif feat.shape[1]!=gh*gw:
        raise RuntimeError(f"patch count mismatch: feat {feat.shape} vs grid {gh}x{gw}={gh*gw}")
    f=F.normalize(feat[0], dim=-1)        # (gh*gw,384)
    return f, (gh,gw)

def patch_purity_mask(label_key_2d, gh, gw, onto):
    """Per-patch mapping: pure-ID = the patch block contains only id_* pixels (no OOD/ignore) -> True (eligible for the bank). Returns a (gh*gw,) bool.
    Uses adaptive_max_pool2d to output exactly gh x gw, without requiring the original size to be divisible (avoids index out-of-bounds)."""
    test_lut=np.full(256,np.float32(IGNORE),np.float32)
    for k,v in onto["test_lut"].items(): test_lut[k]=float(v)
    role=test_lut[label_key_2d]           # 0=ID, 1=OOD, 255=ign  (HxW float)
    r=torch.from_numpy(role)[None,None]   # 1x1xHxW
    rmax=F.adaptive_max_pool2d(r,(gh,gw))[0,0].numpy()   # max within block; ==0 => all ID
    pure = (rmax < 0.5)                    # block max < 0.5 (i.e. all 0 = ID) => pure-ID patch
    return pure.reshape(-1)

def build_bank(max_bank=1_000_000, per_frame=400, seed=0, dataset="goose"):
    m,mean,std=load_model()
    DS,onto_loader=get_dataset(dataset); onto=onto_loader()
    ds=DS("train", mode="test")           # test_lut provides ID/OOD
    rng=np.random.default_rng(seed); feats=[]
    t0=time.time(); ntot=0
    for i in range(len(ds)):
        s=ds[i]; f,(gh,gw)=extract_patches(m,mean,std,s["image"])
        pure=patch_purity_mask(s["label_key"], gh, gw, onto)
        idx=np.nonzero(pure)[0]
        if len(idx)>per_frame: idx=rng.choice(idx,per_frame,replace=False)
        if len(idx): feats.append(f[idx].cpu()); ntot+=len(idx)
        if i%200==0: print(f"  bank {i}/{len(ds)} acc={ntot} ({time.time()-t0:.0f}s)", flush=True)
        if ntot>=max_bank: break
    bank=torch.cat(feats)[:max_bank].contiguous()
    bp=bank_path(dataset); torch.save(bank, bp)
    print(f"[bank] {bank.shape} saved {bp} ({time.time()-t0:.0f}s)")
    return bank

@torch.no_grad()
def score_frame(m,mean,std,bank,img_np,k=1):
    f,(gh,gw)=extract_patches(m,mean,std,img_np)        # (gh*gw,384)
    # nearest-neighbor cosine distance = 1 - max cos. Computed in chunks to avoid OOM
    sims=torch.empty(f.shape[0], device=DEV)
    B=2048
    bankT=bank.T                                         # bank is already on DEV; .T is a view, no copy
    for s in range(0,f.shape[0],B):
        chunk=f[s:s+B]                                   # (b,384)
        sim=chunk @ bankT                                # (b, |bank|)
        sims[s:s+B]=sim.max(dim=1).values
        del sim
    score_patch=(1-sims).reshape(gh,gw)                  # anomaly score: larger the farther from the bank
    # upsample back to the original resolution (1000x2048)
    up=F.interpolate(score_patch[None,None], size=img_np.shape[:2], mode="bilinear", align_corners=False)[0,0]
    return up.cpu().numpy()

def smoke():
    """Small CPU check: 2 frames, verifies shape / patch alignment / score upsampling, without building the full bank."""
    m,mean,std=load_model(); onto=load_goose_ontology()
    ds=GooseOOD("train",mode="test")
    s=ds[0]; f,(gh,gw)=extract_patches(m,mean,std,s["image"])
    print(f"feat {tuple(f.shape)} grid {gh}x{gw}={gh*gw} (should be equal)")
    pure=patch_purity_mask(s["label_key"],gh,gw,onto)
    print(f"pure-ID patches: {pure.sum()}/{len(pure)} ({pure.mean()*100:.0f}%)")
    # use the ID patches of 2 frames as a mini-bank, score the 3rd frame, verify score-map shape/range
    bank=f[np.nonzero(pure)[0][:200]]
    sc=score_frame(m,mean,std,bank.to(DEV),ds[2]["image"])
    print(f"score map {sc.shape} (should be =1000x2048) range[{sc.min():.3f},{sc.max():.3f}]")
    print("smoke ✅" if sc.shape==s['image'].shape[:2] and gh*gw==f.shape[0] else "smoke 🔴")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cmd",choices=["build","eval","smoke"])
    ap.add_argument("--split",default="test"); ap.add_argument("--max_bank",type=int,default=1_000_000)
    ap.add_argument("--per_frame",type=int,default=400); ap.add_argument("--limit",type=int,default=None)
    ap.add_argument("--downsample",type=int,default=4)
    ap.add_argument("--dump",action="store_true")
    ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp","goose_v2","goose_v3"])
    a=ap.parse_args()
    print(f"device={DEV} dataset={a.dataset}")
    if a.cmd=="smoke": smoke(); sys.exit(0)
    if a.cmd=="build": build_bank(a.max_bank,a.per_frame,dataset=a.dataset); sys.exit(0)
    if a.cmd=="eval":
        import metrics as M
        DS,_=get_dataset(a.dataset)
        m,mean,std=load_model(); bank=torch.load(bank_path(a.dataset)).to(DEV)
        ds=DS(a.split,mode="test"); N=len(ds) if a.limit is None else min(a.limit,len(ds))
        scores=[];gts=[];keys=[];t0=time.time()
        for i in range(N):
            s=ds[i]; sc=score_frame(m,mean,std,bank,s["image"]); gt=s["target"]; key=s["label_key"]
            if a.downsample>1: sc=sc[::a.downsample,::a.downsample]; gt=gt[::a.downsample,::a.downsample]; key=key[::a.downsample,::a.downsample]
            scores.append(sc.astype(np.float16));gts.append(gt)
            if a.dump: keys.append(key.copy())
            if i%200==0: print(f"  eval {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
        pix=M.pixel_pr_auroc_fpr95(scores,gts)
        pfx="" if a.dataset=="goose" else f"{a.dataset}_"   # keep the original filename for goose for backward compatibility
        res=dict(method="dinov2_knn",dataset=a.dataset,split=a.split,n_frames=N,sec=round(time.time()-t0,1),pixel=pix,bank=list(bank.shape))
        json.dump(res,open(f"{ROOT}/results/dinov2_knn_{pfx}{a.split}.json","w"),indent=2)
        print(f"[dinov2_knn/{a.dataset}/{a.split}] AP={pix['AP']:.4f} AUROC={pix['AUROC']:.4f} FPR95={pix['FPR95']:.4f} (n_pos={pix['n_pos']}) [saved]")
        if a.dump:
            outp=f"{ROOT}/results/dinov2_scores_{pfx}{a.split}.npz"
            np.savez_compressed(outp, scores=np.stack(scores), keys=np.stack(keys), downsample=a.downsample)
            print(f"  [dump] {outp} scores{np.stack(scores).shape} keys{np.stack(keys).shape}")
