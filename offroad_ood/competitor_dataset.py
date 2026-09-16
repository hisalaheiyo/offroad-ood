#!/usr/bin/env python
"""
Competitor-comparison track dataloaders (reproduce the OOD definition + official split of Anomalies-by-Synthesis, for a fair comparison against their published AP).
- RellisCompOOD: RELLIS official train/val/test.lst (note: the official split has leakage, but both sides use the same protocol so the comparison is fair) + competitor ontology (OOD = vehicle/building/person).
- RugdCompOOD:   RUGD official scene split (= ours, leakage-free by construction) + competitor ontology.
The interface is consistent with the other OOD dataloaders.
"""
import os, json, glob, numpy as np
from PIL import Image
from ontology import load_competitor_ontology, IGNORE, ROOT
from rugd_dataset import RugdOOD

def _lut(d):
    a=np.full(256, IGNORE, np.uint8)
    for k,v in d.items(): a[k]=v
    return a

RELRT=f"{ROOT}/data/rellis3d/Rellis-3D"; RELLST=f"{ROOT}/data/rellis3d"

class RellisCompOOD:
    def __init__(self, split="train", mode=None):
        self.onto=load_competitor_ontology("rellis")
        self.mode=mode or ("train" if split=="train" else "test")
        self.train_lut=_lut(self.onto["train_lut"]); self.test_lut=_lut(self.onto["test_lut"])
        self.frames=[]
        for ln in open(f"{RELLST}/{split}.lst"):
            p=ln.split()
            if len(p)<2: continue
            ip=f"{RELRT}/{p[0]}"; lp=f"{RELRT}/{p[1]}"
            if os.path.exists(ip) and os.path.exists(lp): self.frames.append((ip,lp))
        self.frames.sort()
    def __len__(self): return len(self.frames)
    def __getitem__(self,i):
        ip,lp=self.frames[i]
        img=np.asarray(Image.open(ip).convert("RGB"))
        key=np.asarray(Image.open(lp))
        if key.ndim==3: key=key[...,0]
        tgt=self.train_lut[key] if self.mode=="train" else self.test_lut[key]
        return dict(image=img, target=tgt, label_key=key, image_path=ip)

class RugdCompOOD(RugdOOD):
    def __init__(self, split="train", mode=None):
        super().__init__(split, mode)
        self.onto=load_competitor_ontology("rugd")                  # switch to competitor ontology
        self.train_lut=_lut(self.onto["train_lut"]); self.test_lut=_lut(self.onto["test_lut"])

if __name__=="__main__":
    for nm,cls in [("rellis_comp",RellisCompOOD),("rugd_comp",RugdCompOOD)]:
        o=load_competitor_ontology(nm.replace("_comp",""))
        print(f"\n[{nm}] ID classes={o['num_train']} OOD={len(o['ood_keys'])} ignore={len(o['ignore_keys'])} coverage{o['num_train']+len(o['ood_keys'])+len(o['ignore_keys'])}/{len(o['key2name'])}")
        print("  OOD:", [o['info'][k]['name'] for k in o['ood_keys']])
        for sp in ("train","val","test"):
            d=cls(sp); dt=cls(sp,mode="test"); n=len(d)
            anom=tot=0;nanom=0;bad=0;idxs=list(range(0,n,max(1,n//25)))
            for j in idxs:
                t=dt[j]["target"];a=int((t==1).sum());anom+=a;tot+=t.size;nanom+=(a>0)
                v=d[j]["target"];v=v[v!=255]
                if v.size and (v.min()<0 or v.max()>=o['num_train']): bad+=1
            print(f"  {sp}: {n} frames sampled anomaly frames{nanom}/{len(idxs)} anomaly pixels{anom/max(tot,1)*100:.2f}% illegal train_id={bad}")
