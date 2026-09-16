#!/usr/bin/env python
"""
COCO object outlier sampler (for OE): returns a random object's (RGB crop, binary mask), for pasting onto an off-road image as an outlier.
Uses COCO val2017 + instance annotations. Filters out too-small/crowd objects.
"""
import os, numpy as np
from PIL import Image
from pycocotools.coco import COCO

class CocoOutliers:
    def __init__(self, img_dir, ann_file, min_area=3000, seed=0):
        self.coco=COCO(ann_file); self.img_dir=img_dir
        self.ann_ids=[a for a in self.coco.getAnnIds()
                      if (self.coco.anns[a]["area"]>=min_area and not self.coco.anns[a]["iscrowd"])]
        assert self.ann_ids, "no eligible COCO objects"
        self.rng=np.random.default_rng(seed)
        print(f"[CocoOutliers] {len(self.ann_ids)} objects (area>={min_area})",flush=True)

    def sample(self):
        """Returns (crop_rgb HxWx3 uint8, crop_mask HxW bool)."""
        for _ in range(10):
            aid=int(self.ann_ids[self.rng.integers(len(self.ann_ids))])
            ann=self.coco.anns[aid]; info=self.coco.imgs[ann["image_id"]]
            try:
                img=np.array(Image.open(os.path.join(self.img_dir,info["file_name"])).convert("RGB"))
                m=self.coco.annToMask(ann).astype(bool)
            except Exception: continue
            x,y,w,h=[int(v) for v in ann["bbox"]]
            x,y=max(0,x),max(0,y); w,h=max(1,w),max(1,h)
            crop=img[y:y+h, x:x+w]; cm=m[y:y+h, x:x+w]
            if cm.sum()>=500 and crop.shape[0]>=8 and crop.shape[1]>=8:
                return crop, cm
        return crop, cm

def paste_outlier(img, obj_crop, obj_mask, rng, scale_range=(0.08,0.30)):
    """Paste the object onto img at a random position/scale, returns (composite img, outlier_mask HxW bool)."""
    from PIL import Image as PImage
    H,W=img.shape[:2]; oh,ow=obj_crop.shape[:2]
    # target scale: object's longest edge = scale * image's longest edge
    tgt=rng.uniform(*scale_range)*max(H,W)
    sc=tgt/max(oh,ow); nh,nw=max(8,int(oh*sc)),max(8,int(ow*sc))
    if nh>=H or nw>=W: nh,nw=min(nh,H-1),min(nw,W-1)
    oc=np.array(PImage.fromarray(obj_crop).resize((nw,nh),PImage.BILINEAR))
    om=np.array(PImage.fromarray(obj_mask.astype(np.uint8)*255).resize((nw,nh),PImage.NEAREST))>127
    py=int(rng.integers(0,max(1,H-nh))); px=int(rng.integers(0,max(1,W-nw)))
    out=img.copy(); outlier=np.zeros((H,W),bool)
    region=out[py:py+nh, px:px+nw]; region[om]=oc[om]; out[py:py+nh, px:px+nw]=region
    outlier[py:py+nh, px:px+nw][om]=True
    return out, outlier

if __name__=="__main__":  # smoke (requires COCO already extracted)
    import sys
    root=sys.argv[1] if len(sys.argv)>1 else "../data/coco/val2017"
    ann=sys.argv[2] if len(sys.argv)>2 else "../data/coco/annotations/instances_val2017.json"
    co=CocoOutliers(root,ann)
    crop,m=co.sample(); print(f"sample crop{crop.shape} mask sum={m.sum()}")
    img=np.zeros((1000,2048,3),np.uint8)
    comp,ol=paste_outlier(img,crop,m,np.random.default_rng(0))
    print(f"composite{comp.shape} outlier pixels={ol.sum()} ({ol.mean()*100:.2f}%) ✅")
