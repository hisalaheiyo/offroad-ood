#!/usr/bin/env python
"""
S2: fine-tune HuggingFace Mask2Former (mask-transformer) on off-road ID classes, to serve as the RbA modern OOD baseline.
- Training uses mode='train' semantic target (train_id 0..NC-1, OOD/ignore=255) -> OOD pixels are ignored during training, leakage-free (same as the linear head).
- Monitor val mIoU to ensure convergence (otherwise an RbA failure would be confounded as a 'weak segmenter' rather than a base-rate problem).
Correctness reflection:
  (1) num_labels=NC + ignore_mismatched_sizes -> reset the cityscapes (19) class head to our NC classes.
  (2) processor(segmentation_maps, ignore_index=255) -> mask_labels/class_labels, 255 automatically excluded.
  (3) Skip frames with no ID pixels (empty class_labels would raise an error).
  (4) mIoU computed only over ID classes (OOD/ignore=255 excluded).
Usage (GPU): python m2f_train.py --dataset goose --epochs 8
"""
import os, sys, time, argparse, numpy as np, torch
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import ROOT
from datasets import get_dataset
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "facebook/mask2former-swin-small-cityscapes-semantic"
SHORT = 512   # training short edge (controls GPU memory/speed); GOOSE 1920x1080 -> ~910x512

def _sfx(seed): return "" if seed == 0 else f"_s{seed}"                 # seed=0=original path (backward compatible)
def ckpt_path(ds, seed=0): return f"{ROOT}/results/m2f_{ds}{_sfx(seed)}.pt"
def meta_path(ds, seed=0): return f"{ROOT}/results/m2f_{ds}{_sfx(seed)}_meta.json"

def make_proc(short=SHORT):
    p = Mask2FormerImageProcessor.from_pretrained(CKPT)
    p.ignore_index = 255
    p.do_reduce_labels = False
    p.size = {"shortest_edge": int(short), "longest_edge": max(1333, int(short) * 2)}
    return p

def load_short(ds):
    """eval reads the training-time resolution, to prevent train/eval res mismatch."""
    import json as _j, os
    try: return _j.load(open(meta_path(ds)))["short"]
    except Exception: return SHORT

def load_semantic(ds_obj, i):
    """Returns (img HxWx3 uint8, sem HxW int64 with 255=ignore/OOD)."""
    s = ds_obj[i]
    return s["image"], s["target"].astype(np.int64)

@torch.no_grad()
def eval_miou(model, proc, DS, NC, n_max=80):
    """val ID-class mIoU (255 excluded). Convergence monitoring."""
    model.eval()
    dsv = DS("val", mode="train")   # train mode gives semantic GT
    inter = np.zeros(NC); union = np.zeros(NC)
    N = min(len(dsv), n_max)
    for i in range(N):
        img, sem = load_semantic(dsv, i)
        enc = proc(images=[img], return_tensors="pt")
        px = enc["pixel_values"].to(DEV)
        out = model(pixel_values=px)
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[sem.shape])[0].cpu().numpy()
        valid = sem != 255
        for c in range(NC):
            pc = (seg == c) & valid; gc = (sem == c) & valid
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum()
    iou = inter / np.maximum(union, 1)
    present = union > 0
    return float(iou[present].mean()), int(present.sum())

def main(dataset="goose", epochs=8, lr=5e-5, bs=2, smoke=False, short=SHORT, seed=0):
    import json as _j
    torch.manual_seed(seed); np.random.seed(seed)                       # seed robustness: class-head init + data order
    DS, onto_loader = get_dataset(dataset); onto = onto_loader(); NC = onto["num_train"]
    print(f"[m2f] dataset={dataset} NC={NC} short={short} seed={seed} dev={DEV}", flush=True)
    proc = make_proc(short)
    if not smoke: _j.dump({"short": int(short)}, open(meta_path(dataset, seed), "w"))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        CKPT, num_labels=NC, ignore_mismatched_sizes=True).to(DEV)
    ds = DS("train", mode="train"); N = len(ds)

    if smoke:
        # single-step verification of the data->loss pipeline
        imgs, sems = [], []
        for i in range(2):
            img, sem = load_semantic(ds, i)
            if (sem != 255).sum() == 0: continue
            imgs.append(img); sems.append(sem)
        enc = proc(images=imgs, segmentation_maps=sems, return_tensors="pt")
        enc = {k: (v.to(DEV) if torch.is_tensor(v) else [x.to(DEV) for x in v]) for k, v in enc.items()}
        out = model(pixel_values=enc["pixel_values"], mask_labels=enc["mask_labels"], class_labels=enc["class_labels"])
        print(f"smoke loss={out.loss.item():.4f} class_labels[0]={enc['class_labels'][0].tolist()[:8]}")
        print(f"  mask_labels[0].shape={tuple(enc['mask_labels'][0].shape)}")
        miou, npr = eval_miou(model, proc, DS, NC, n_max=6)
        print(f"  pretrained (not fine-tuned) val mIoU={miou:.3f} on {npr} classes (should be low, rises after fine-tuning)")
        print("smoke ✅" if torch.isfinite(out.loss) else "smoke 🔴")
        return

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time(); best = -1
    for ep in range(epochs):
        model.train(); order = np.random.permutation(N); tot = 0; nb = 0
        for j in range(0, N, bs):
            idx = order[j:j+bs]; imgs, sems = [], []
            for ii in idx:
                img, sem = load_semantic(ds, int(ii))
                if (sem != 255).sum() == 0: continue         # skip frames with no ID pixels
                imgs.append(img); sems.append(sem)
            if not imgs: continue
            enc = proc(images=imgs, segmentation_maps=sems, return_tensors="pt")
            enc = {k: (v.to(DEV) if torch.is_tensor(v) else [x.to(DEV) for x in v]) for k, v in enc.items()}
            out = model(pixel_values=enc["pixel_values"], mask_labels=enc["mask_labels"], class_labels=enc["class_labels"])
            loss = out.loss
            if torch.isfinite(loss):
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); tot += loss.item(); nb += 1
            if j % (bs*200) == 0:
                print(f"  ep{ep} {j}/{N} loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        miou, npr = eval_miou(model, proc, DS, NC)
        print(f"epoch{ep} mean_loss={tot/max(nb,1):.4f} val_mIoU={miou:.4f}({npr} classes) ({time.time()-t0:.0f}s)", flush=True)
        if miou > best:
            best = miou; torch.save(model.state_dict(), ckpt_path(dataset, seed))
            print(f"  [saved best] {ckpt_path(dataset, seed)} mIoU={best:.4f}", flush=True)
    print(f"[done] best val_mIoU={best:.4f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="goose", choices=["goose", "rellis", "wildscenes", "rugd", "rugd_comp"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--short", type=int, default=SHORT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.dataset, a.epochs, a.lr, a.bs, a.smoke, a.short, a.seed)
