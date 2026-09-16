#!/usr/bin/env python
"""
Leakage ablation (A3, controlled): shows that "same-scene leakage in the official split inflates performance, and only our leakage-free split is realistic".
Controlled: the same batch of eval frames (half-A of the test scenes), varying only whether the bank contains same-scene background:
  CLEAN bank = ID-patches from train scenes only
  LEAKY bank = train scenes + half-B of the test scenes (different frames from half-A but same scenes = simulates official-split leakage)
Both banks have the same size 1M (fair). If LEAKY has lower FPR95 / higher AUROC = leakage inflates performance (core benchmark argument).
Usage (GPU): python leakage_ablation.py
"""
import json, time, numpy as np, torch
from PIL import Image
from ontology import load_goose_ontology, IGNORE, ROOT, _eff_role  # noqa
import dinov2_baseline as D
from goose_dataset import list_frames
import metrics as M

def site_of(scen):
    SITES=["garching","aying","neubiberg","campus","siegertsbrunn","hoehenkirchner","putzbrunn","solalinden","hohenbrunn","neuperlach","flight"]
    for s in SITES:
        if s in scen: return s
    return "?"

@torch.no_grad()
def sample_patches(m,mean,std,onto,frame_list,target,per_frame,seed=0):
    """Sample target pure ID-patch features from frame_list (shuffle frames first to avoid ordering bias). Returns a CPU tensor."""
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(frame_list)); feats=[]; ntot=0
    for j in order:
        img_p,lab_p=frame_list[j]
        img=np.asarray(Image.open(img_p).convert("RGB")); key=np.asarray(Image.open(lab_p))
        f,(gh,gw)=D.extract_patches(m,mean,std,img)
        pure=D.patch_purity_mask(key,gh,gw,onto); idx=np.nonzero(pure)[0]
        if len(idx)>per_frame: idx=rng.choice(idx,per_frame,replace=False)
        if len(idx): feats.append(f[idx].cpu()); ntot+=len(idx)
        if ntot>=target: break
    return torch.cat(feats)[:target] if feats else torch.empty(0,384)

@torch.no_grad()
def eval_on(m,mean,std,bank,test_lut,frame_list,downsample=4):
    scores=[];gts=[]
    for img_p,lab_p in frame_list:
        img=np.asarray(Image.open(img_p).convert("RGB")); key=np.asarray(Image.open(lab_p))
        sc=D.score_frame(m,mean,std,bank,img)[::downsample,::downsample].astype(np.float16)
        gt=test_lut[key][::downsample,::downsample]
        scores.append(sc); gts.append(gt)
    return M.pixel_pr_auroc_fpr95(scores,gts)

def main():
    onto=load_goose_ontology(); m,mean,std=D.load_model()
    test_lut=np.full(256,IGNORE,np.uint8)
    for k,v in onto["test_lut"].items(): test_lut[k]=v
    sp=json.load(open(f"{ROOT}/configs/goose_site_split.json"))
    fbs=list_frames()
    train_frames=[]; testA=[]; testB=[]
    for scen,frs in fbs.items():
        st=site_of(scen)
        if st in sp["train"]: train_frames+=frs
        elif st in sp["test"]:
            for i,f in enumerate(sorted(frs)): (testA if i%2==0 else testB).append(f)
    print(f"train frames{len(train_frames)} testA{len(testA)} testB{len(testB)}",flush=True)
    t0=time.time()
    # CLEAN=1M train; LEAKY=500k train+500k testB (explicitly controlled to ensure testB actually enters the bank)
    clean=sample_patches(m,mean,std,onto,train_frames,1_000_000,400).contiguous().to(D.DEV)
    print(f"clean bank{tuple(clean.shape)} (all train) ({time.time()-t0:.0f}s)",flush=True)
    tb_train=sample_patches(m,mean,std,onto,train_frames,500_000,400,seed=1)
    tb_test =sample_patches(m,mean,std,onto,testB,500_000,800,seed=2)
    leaky=torch.cat([tb_train,tb_test]).contiguous().to(D.DEV)
    print(f"leaky bank{tuple(leaky.shape)} (train{tb_train.shape[0]}+testB{tb_test.shape[0]}=same-scene{tb_test.shape[0]/leaky.shape[0]*100:.0f}%) ({time.time()-t0:.0f}s)",flush=True)
    rc=eval_on(m,mean,std,clean,test_lut,testA); print(f"CLEAN(leakage-free): AP{rc['AP']:.4f} AUROC{rc['AUROC']:.4f} FPR95{rc['FPR95']:.4f}",flush=True)
    rl=eval_on(m,mean,std,leaky,test_lut,testA); print(f"LEAKY(with same-scene): AP{rl['AP']:.4f} AUROC{rl['AUROC']:.4f} FPR95{rl['FPR95']:.4f}",flush=True)
    print(f"Δ(leaky-clean): AP{rl['AP']-rc['AP']:+.4f} AUROC{rl['AUROC']-rc['AUROC']:+.4f} FPR95{rl['FPR95']-rc['FPR95']:+.4f}")
    print(f"  → FPR95 down / AUROC up = leakage inflated performance = only the leakage-free split reflects reality (benchmark argument)")
    json.dump(dict(clean=rc,leaky=rl,n_testA=len(testA),n_testB=len(testB)),open(f"{ROOT}/results/leakage_ablation.json","w"),indent=2)
    print("[saved] results/leakage_ablation.json")

if __name__=="__main__": main()
