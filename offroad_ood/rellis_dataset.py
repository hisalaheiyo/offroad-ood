#!/usr/bin/env python
"""
RELLIS-3D dataloader (pixel-masking, leakage-free sequence-level split).
The official split has severe leakage (sequence 00000 appears in train/val/test) → use configs/rellis_seq_split.json with disjoint sequences instead.
Images: data/rellis3d/Rellis-3D/<seq>/pylon_camera_node/<frame>.jpg
Labels: .../<seq>/pylon_camera_node_label_id/<frame>.png (sparse ids, see meta_ontology RELLIS-3D)
Interface consistent with GooseOOD/WildScenesOOD: __getitem__->dict(image,target,label_key,image_path).
"""
import os, json, glob, numpy as np
from PIL import Image
from ontology import load_rellis_ontology, IGNORE, ROOT

RT = f"{ROOT}/data/rellis3d/Rellis-3D"
SPLIT = f"{ROOT}/configs/rellis_seq_split.json"

def _lut(d):
    a=np.full(256, IGNORE, np.uint8)
    for k,v in d.items(): a[k]=v
    return a

class RellisOOD:
    def __init__(self, split="train", mode=None):
        self.onto=load_rellis_ontology()
        self.mode=mode or ("train" if split=="train" else "test")
        self.train_lut=_lut(self.onto["train_lut"]); self.test_lut=_lut(self.onto["test_lut"])
        seqs=json.load(open(SPLIT))[split]
        self.frames=[]
        for s in seqs:
            for lp in sorted(glob.glob(f"{RT}/{s}/pylon_camera_node_label_id/*.png")):
                fn=os.path.basename(lp).replace(".png",".jpg")
                ip=f"{RT}/{s}/pylon_camera_node/{fn}"
                if os.path.exists(ip): self.frames.append((ip,lp))
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
    o=load_rellis_ontology()
    print(f"[RELLIS] ID classes={o['num_train']} OOD={len(o['ood_keys'])} ignore={len(o['ignore_keys'])} coverage{o['num_train']+len(o['ood_keys'])+len(o['ignore_keys'])}/{len(o['key2name'])}")
    print("ID:", [o['info'][k]['name'] for k in o['id_keys']])
    print("OOD:", [(o['info'][k]['name'],o['info'][k]['tier']) for k in o['ood_keys']])
    for sp in ("train","val","test"):
        d=RellisOOD(sp); dt=RellisOOD(sp,mode="test"); n=len(d)
        anom=tot=0; nanom=0; bad=0
        for j in range(0,n,max(1,n//30)):
            t=dt[j]["target"]
            a=int((t==1).sum()); anom+=a; tot+=t.size; nanom+=(a>0)
            # train_id validity
            ttr=d[j]["target"]; v=ttr[ttr!=255]
            if v.size and (v.min()<0 or v.max()>=o['num_train']): bad+=1
        print(f"  {sp}: {n} frames, sampled anomaly frames {nanom}/{len(range(0,n,max(1,n//30)))} anomaly pixels {anom/max(tot,1)*100:.2f}% invalid train_id frames={bad}")
