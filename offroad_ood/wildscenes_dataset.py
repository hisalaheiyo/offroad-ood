#!/usr/bin/env python
"""
WildScenes dataloader (pixel-masking, official 45m-buffer leakage-free split)
split csv: data/wildscenes/splits/{train,val,test}.csv, each row: ts, WildScenes2d/<seq>/image/<ts>.png, .../indexLabel/<ts>.png
indexLabel: mode L uint8, values 0-18 (=label_key, see meta_ontology WildScenes); pixel-masking same as GOOSE.
"""
import os, csv, numpy as np
from PIL import Image
from ontology import load_wildscenes_ontology, IGNORE, ROOT

WS = f"{ROOT}/data/wildscenes"

def _lut(d):
    a=np.full(256, IGNORE, np.uint8)
    for k,v in d.items(): a[k]=v
    return a

class WildScenesOOD:
    def __init__(self, split="train", mode=None):
        self.onto=load_wildscenes_ontology()
        self.mode=mode or ("train" if split=="train" else "test")
        self.train_lut=_lut(self.onto["train_lut"]); self.test_lut=_lut(self.onto["test_lut"])
        self.frames=[]
        with open(f"{WS}/splits/{split}.csv") as f:
            for row in csv.reader(f):
                if len(row)<3 or row[1].strip()=="" or "image" not in row[1]: continue
                img=f"{WS}/{row[1].strip()}"; lab=f"{WS}/{row[2].strip()}"
                if os.path.exists(img) and os.path.exists(lab): self.frames.append((img,lab))
        self.frames.sort()
    def __len__(self): return len(self.frames)
    def __getitem__(self,i):
        ip,lp=self.frames[i]
        img=np.asarray(Image.open(ip).convert("RGB"))
        key=np.asarray(Image.open(lp))
        if key.ndim==3: key=key[...,0]
        tgt=self.train_lut[key] if self.mode=="train" else self.test_lut[key]
        return dict(image=img, target=tgt, label_key=key, image_path=ip)

if __name__=="__main__":
    for sp in ("train","val","test"):
        d=WildScenesOOD(sp); n=len(d)
        dt=WildScenesOOD(sp,mode="test")
        anom=idp=ign=tot=0; nanom=0
        for j in range(0,n,max(1,n//40)):
            t=dt[j]["target"]; a=(t==1).sum()
            anom+=a; idp+=(t==0).sum(); ign+=(t==255).sum(); tot+=t.size
            if a>=10: nanom+=1
        tr=WildScenesOOD(sp,mode="train")[0]["target"]; u=np.unique(tr); bad=u[(u>=12)&(u!=255)]
        print(f"[{sp:5s}] {n} frames | OOD{anom/tot*100:.1f}% ID{idp/tot*100:.1f}% ign{ign/tot*100:.1f}% with-anomaly samples~{nanom} | train_id range{u.min()}-{u[u!=255].max() if (u!=255).any() else 'NA'} illegal{bad.tolist() if len(bad) else 'none✅'}")
