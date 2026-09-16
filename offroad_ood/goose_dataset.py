#!/usr/bin/env python
"""
OffRoad-OOD — GOOSE dataloader (pixel-masking, framework-agnostic core + optional torch wrapper)
- split from configs/goose_site_split.json (site-level leakage-free); physical files from the official train/val directories
  (we merge all scenes from official train+val, then re-partition by site into our train/val/test)
- training mode: label = label_key through train_lut -> train_id (0-19) or IGNORE (255); OOD pixels are masked to IGNORE
- test mode: anomaly GT = test_lut -> 1 (OOD) / 0 (ID) / 255 (ignore)
Returns numpy; GooseTorch is an optional torch.utils.data.Dataset wrapper.
"""
import os, glob, json, numpy as np
from PIL import Image
from ontology import load_goose_ontology, IGNORE, ROOT

GOOSE = f"{ROOT}/data/goose/extracted"

def _build_lut_array(lut_dict):
    arr = np.full(256, IGNORE, dtype=np.uint8)
    for k,v in lut_dict.items(): arr[k] = v
    return arr

def list_frames():
    """scenario -> [(image_path, label_path), ...]  scans all scenes in official train+val."""
    out = {}
    for split in ("train","val"):
        for lab in sorted(glob.glob(f"{GOOSE}/labels/{split}/*/*_labelids.png")):
            scen = os.path.basename(os.path.dirname(lab))
            base = os.path.basename(lab).replace("_labelids.png","")
            img = f"{GOOSE}/images/{split}/{scen}/{base}_windshield_vis.png"
            if os.path.exists(img):
                out.setdefault(scen, []).append((img, lab))
    return out

def scenarios_for_site(site, all_scen):
    return [s for s in all_scen if site in s]

class GooseOOD:
    SPLIT_FILE="goose_site_split.json"
    """Framework-agnostic core. mode in {'train','test'}; split in {'train','val','test'} (our site-level split)."""
    def __init__(self, split="train", mode=None):
        self.onto = load_goose_ontology()
        self.split = split
        self.mode = mode or ("train" if split=="train" else "test")
        self.train_lut = _build_lut_array(self.onto["train_lut"])
        self.test_lut  = _build_lut_array(self.onto["test_lut"])
        site_split = json.load(open(f"{ROOT}/configs/"+getattr(self,"SPLIT_FILE","goose_site_split.json")))
        sites = site_split[split]
        frames_by_scen = list_frames()
        all_scen = list(frames_by_scen)
        self.frames = []
        for site in sites:
            for scen in scenarios_for_site(site, all_scen):
                self.frames += frames_by_scen[scen]
        self.frames.sort()
    def __len__(self): return len(self.frames)
    def __getitem__(self, i):
        img_p, lab_p = self.frames[i]
        img = np.asarray(Image.open(img_p).convert("RGB"))          # HxWx3 uint8
        key = np.asarray(Image.open(lab_p))                          # HxW uint8 (label_key)
        if self.mode == "train":
            target = self.train_lut[key]      # train_id or 255 (OOD masked)
        else:
            target = self.test_lut[key]       # 1/0/255 anomaly GT
        return dict(image=img, target=target, label_key=key, image_path=img_p)

if __name__ == "__main__":
    # self-check: sizes of the three splits + train/test target values + consistency with audit
    for sp in ("train","val","test"):
        d = GooseOOD(sp)
        n = len(d)
        s = d[0]
        # sampled statistics of anomaly fraction (test mode)
        dt = GooseOOD(sp, mode="test")
        anom=idp=ign=0; tot=0; nanom_frames=0
        for j in range(0, n, max(1,n//40)):   # sample ~40 frames
            t = dt[j]["target"]
            a=(t==1).sum(); ip=(t==0).sum(); ig=(t==255).sum()
            anom+=a; idp+=ip; ign+=ig; tot+=t.size
            if a>=10: nanom_frames+=1
        print(f"[{sp:5s}] {n} frames | sampled: OOD{anom/tot*100:.1f}% ID{idp/tot*100:.1f}% ign{ign/tot*100:.1f}% | sampled frames with anomalies~{nanom_frames}")
        # validation of training target values
        tr = GooseOOD(sp, mode="train")[0]["target"]
        u = np.unique(tr); bad = u[(u>=20)&(u!=255)]
        print(f"        train target unique range: {u.min()}-{u[u!=255].max() if (u!=255).any() else 'NA'} +IGNORE | invalid values{bad.tolist() if len(bad) else 'none✅'} | image shape{tr.shape}")


class GooseV2OOD(GooseOOD):
    SPLIT_FILE="goose_site_split_v2.json"

class GooseV3OOD(GooseOOD):
    SPLIT_FILE="goose_site_split_v3.json"
