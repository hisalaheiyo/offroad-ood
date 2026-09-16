#!/usr/bin/env python
"""
#5 Architecture control: train SegFormer (per-pixel transformer, end-to-end) on off-road ID classes, scored with MSP/MaxLogit/Energy.
Comparison: MSP-per-pixel (frozen DINOv2 + linear head, weak AP) vs SegFormer-MSP (end-to-end per-pixel) vs Mask2Former-MSP (mask, strong AP).
  If SegFormer-MSP ≈ Mask2Former → the gain comes from "end-to-end training" (per-pixel is enough, training is needed).
  If SegFormer-MSP << Mask2Former → the gain comes from the mask-decoder architecture.
OOD pixels are set to 255 ignore during training (SegFormer semantic_loss_ignore_index defaults to 255, leakage-free). Monitor val mIoU.
Usage (GPU): python segformer_train.py --dataset goose --epochs 10
"""
import os, time, argparse, json, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import ROOT
from datasets import get_dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
SHORT = 512

def ckpt_path(ds): return f"{ROOT}/results/segformer_{ds}.pt"

def make_proc(short=SHORT):
    p = SegformerImageProcessor.from_pretrained(CKPT)
    p.do_reduce_labels = False
    p.size = {"height": int(short), "width": int(short)}   # SegFormer square processing
    return p

@torch.no_grad()
def logits_full(model, proc, img):
    H, W = img.shape[:2]
    px = proc(images=[img], return_tensors="pt")["pixel_values"].to(DEV)
    lo = model(pixel_values=px).logits                      # (1,NC,h/4,w/4)
    return F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False)[0]  # (NC,H,W)

@torch.no_grad()
def eval_miou(model, proc, DS, NC, n_max=80):
    model.eval(); dsv = DS("val", mode="train")
    inter = np.zeros(NC); union = np.zeros(NC); N = min(len(dsv), n_max)
    for i in range(N):
        s = dsv[i]; sem = s["target"].astype(np.int64)
        seg = logits_full(model, proc, s["image"]).argmax(0).cpu().numpy()
        valid = sem != 255
        for c in range(NC):
            pc = (seg == c) & valid; gc = (sem == c) & valid
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum()
    iou = inter / np.maximum(union, 1); pres = union > 0
    return float(iou[pres].mean()), int(pres.sum())

def main(dataset="goose", epochs=10, lr=6e-5, bs=4, short=SHORT, smoke=False):
    DS, ol = get_dataset(dataset); onto = ol(); NC = onto["num_train"]
    print(f"[segformer] {dataset} NC={NC} short={short} dev={DEV}", flush=True)
    proc = make_proc(short)
    model = SegformerForSemanticSegmentation.from_pretrained(CKPT, num_labels=NC, ignore_mismatched_sizes=True).to(DEV)
    model.config.semantic_loss_ignore_index = 255
    ds = DS("train", mode="train"); N = len(ds)
    if smoke:
        imgs, sems = [], []
        for i in range(2):
            s = ds[i]; sem = s["target"].astype(np.int64)
            if (sem != 255).sum() == 0: continue
            imgs.append(s["image"]); sems.append(sem)
        enc = proc(images=imgs, segmentation_maps=sems, return_tensors="pt")
        out = model(pixel_values=enc["pixel_values"].to(DEV), labels=enc["labels"].to(DEV))
        print(f"smoke loss={out.loss.item():.4f} labels uniq={np.unique(sems[0]).tolist()[:8]}")
        print("smoke ✅" if torch.isfinite(out.loss) else "smoke 🔴"); return
    json.dump({"short": int(short)}, open(f"{ROOT}/results/segformer_{dataset}_meta.json", "w"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time(); best = -1
    for ep in range(epochs):
        model.train(); order = np.random.permutation(N); tot = 0; nb = 0
        for j in range(0, N, bs):
            idx = order[j:j+bs]; imgs, sems = [], []
            for ii in idx:
                s = ds[int(ii)]; sem = s["target"].astype(np.int64)
                if (sem != 255).sum() == 0: continue
                imgs.append(s["image"]); sems.append(sem)
            if not imgs: continue
            enc = proc(images=imgs, segmentation_maps=sems, return_tensors="pt")
            out = model(pixel_values=enc["pixel_values"].to(DEV), labels=enc["labels"].to(DEV))
            if torch.isfinite(out.loss):
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += out.loss.item(); nb += 1
            if j % (bs*200) == 0: print(f"  ep{ep} {j}/{N} loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        miou, npr = eval_miou(model, proc, DS, NC)
        print(f"epoch{ep} loss={tot/max(nb,1):.4f} val_mIoU={miou:.4f}({npr} classes) ({time.time()-t0:.0f}s)", flush=True)
        if miou > best: best = miou; torch.save(model.state_dict(), ckpt_path(dataset)); print(f"  [saved best] mIoU={best:.4f}", flush=True)
    print(f"[done] best val_mIoU={best:.4f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose", choices=["goose","rellis","wildscenes","rugd"])
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--short", type=int, default=SHORT)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    main(a.dataset, a.epochs, a.lr, a.bs, a.short, a.smoke)
