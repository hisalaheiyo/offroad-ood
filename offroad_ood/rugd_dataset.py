#!/usr/bin/env python
"""
RUGD dataloader (pixel-masking, official scene-level split is naturally leakage-free).
Annotations = color-encoded (RGB); convert to id via colormap (RUGD_annotation-colormap.txt), then pass through ontology.
Images: data/rugd/RUGD_frames-with-annotations/<scene>/<frame>.png (RGB camera image)
Annotations: data/rugd/RUGD_annotations/<scene>/<frame>.png (RGB color labels, same filename)
Interface consistent with the other OOD dataloaders: __getitem__->dict(image,target,label_key,image_path).
"""
import os, json, glob, numpy as np
from PIL import Image
from ontology import load_rugd_ontology, IGNORE, ROOT

RT = f"{ROOT}/data/rugd"
IMGD = f"{RT}/RUGD_frames-with-annotations"
ANND = f"{RT}/RUGD_annotations"
SPLIT = f"{ROOT}/configs/rugd_scene_split.json"
CMAP = f"{ANND}/RUGD_annotation-colormap.txt"

def _load_colormap():
    """RUGD_annotation-colormap.txt: 'id name r g b' -> {(r,g,b): id}."""
    cm={}
    for ln in open(CMAP):
        p=ln.split()
        if len(p)>=5:
            cid=int(p[0]); r,g,b=int(p[-3]),int(p[-2]),int(p[-1]); cm[(r,g,b)]=cid
    return cm

def _lut(d):
    a=np.full(256, IGNORE, np.uint8)
    for k,v in d.items(): a[k]=v
    return a

class RugdOOD:
    def __init__(self, split="train", mode=None):
        self.onto=load_rugd_ontology()
        self.mode=mode or ("train" if split=="train" else "test")
        self.train_lut=_lut(self.onto["train_lut"]); self.test_lut=_lut(self.onto["test_lut"])
        self.cmap=_load_colormap()
        # pre-encode colormap: r*65536+g*256+b -> id
        self.enc2id={r*65536+g*256+b:cid for (r,g,b),cid in self.cmap.items()}
        scenes=json.load(open(SPLIT))[split]
        self.frames=[]
        for s in scenes:
            for ap in sorted(glob.glob(f"{ANND}/{s}/*.png")):
                fn=os.path.basename(ap); ip=f"{IMGD}/{s}/{fn}"
                if os.path.exists(ip): self.frames.append((ip,ap))
        self.frames.sort()
    def _rgb2id(self, ann):
        enc=ann[:,:,0].astype(np.uint32)*65536 + ann[:,:,1].astype(np.uint32)*256 + ann[:,:,2].astype(np.uint32)
        key=np.zeros(ann.shape[:2], np.uint8)
        for e,cid in self.enc2id.items(): key[enc==e]=cid
        return key
    def __len__(self): return len(self.frames)
    def __getitem__(self,i):
        ip,ap=self.frames[i]
        img=np.asarray(Image.open(ip).convert("RGB"))
        ann=np.asarray(Image.open(ap).convert("RGB"))
        key=self._rgb2id(ann)
        tgt=self.train_lut[key] if self.mode=="train" else self.test_lut[key]
        return dict(image=img, target=tgt, label_key=key, image_path=ip)

if __name__=="__main__":
    o=load_rugd_ontology()
    print(f"[RUGD] ID classes={o['num_train']} OOD={len(o['ood_keys'])} ignore={len(o['ignore_keys'])} coverage{o['num_train']+len(o['ood_keys'])+len(o['ignore_keys'])}/{len(o['key2name'])}")
    print("ID:", [o['info'][k]['name'] for k in o['id_keys']])
    print("OOD:", [(o['info'][k]['name'],o['info'][k]['tier']) for k in o['ood_keys']])
    for sp in ("train","val","test"):
        d=RugdOOD(sp); dt=RugdOOD(sp,mode="test"); n=len(d)
        anom=tot=0; nanom=0; bad=0; idxs=list(range(0,n,max(1,n//30)))
        for j in idxs:
            t=dt[j]["target"]; a=int((t==1).sum()); anom+=a; tot+=t.size; nanom+=(a>0)
            ttr=d[j]["target"]; v=ttr[ttr!=255]
            if v.size and (v.min()<0 or v.max()>=o['num_train']): bad+=1
        print(f"  {sp}: {n} frames, sampled anomaly frames {nanom}/{len(idxs)} anomaly pixels {anom/max(tot,1)*100:.2f}% invalid train_id frames={bad}")
