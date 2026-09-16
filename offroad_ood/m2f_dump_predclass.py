#!/usr/bin/env python
"""
Conformal prototype step1: dump Mask2Former per-pixel [predicted terrain class + MSP-M2F score + keys] to the new file m2f_predclass_*.
Used for terrain-conditional conformal operating-point calibration (set per-terrain thresholds from val-ID -> seed-stable + FPR guarantee).
⚠️ Does not overwrite any existing results: only writes the new file m2f_predclass_{pfx}{split}{ssfx}.npz. Reuses the already-trained checkpoint (no retraining).
Correctness notes:
  (1) Predicted class = argmax_c L_c (semantic logit), computed at model resolution, **nearest upsampling** (class indices must not be blended by bilinear) -> [::4,::4] aligned with keys.
  (2) MSP-M2F = -max_c L_c, bilinear upsampling (consistent with existing m2f_rba). predclass and score from the same forward pass -> self-consistent.
  (3) Only loads the checkpoint for inference, does not change any training / existing npz.
Usage (GPU): python m2f_dump_predclass.py --dataset goose --split test --seed 0
"""
import os, time, argparse, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import ROOT
from datasets import get_dataset
from transformers import Mask2FormerForUniversalSegmentation
from m2f_train import CKPT, make_proc, ckpt_path, load_short
DEV = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def pred_and_score(model, proc, img, NC):
    H, W = img.shape[:2]
    enc = proc(images=[img], return_tensors="pt"); px = enc["pixel_values"].to(DEV)
    out = model(pixel_values=px)
    cls = out.class_queries_logits[0]; msk = out.masks_queries_logits[0]
    probs = cls.softmax(-1)[:, :NC]; mprob = msk.sigmoid()
    L = torch.einsum("qc,qhw->chw", probs, mprob)            # (C,h,w)
    pred = L.argmax(0).float()[None, None]                   # (1,1,h,w) predicted class
    msp = (-L.max(0).values)[None, None]                     # (1,1,h,w) MSP-M2F
    pred_up = F.interpolate(pred, size=(H, W), mode="nearest")[0, 0].cpu().numpy().astype(np.uint8)   # nearest!
    msp_up = F.interpolate(msp, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return pred_up, msp_up

def main(dataset="goose", split="test", seed=0):
    DS, ol = get_dataset(dataset); onto = ol(); NC = onto["num_train"]
    proc = make_proc(load_short(dataset))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT, num_labels=NC, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path(dataset, seed), map_location="cpu")); model.to(DEV).eval()
    print(f"[predclass] {dataset} {split} seed={seed} NC={NC} loading{ckpt_path(dataset, seed)}", flush=True)
    dst = DS(split, mode="test"); N = len(dst)
    preds, msps, keys = [], [], []; t0 = time.time()
    for i in range(N):
        s = dst[i]; pr, ms = pred_and_score(model, proc, s["image"], NC)
        preds.append(pr[::4, ::4].copy()); msps.append(ms[::4, ::4].astype(np.float16)); keys.append(s["label_key"][::4, ::4].copy())
        if i % 200 == 0: print(f"  {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    pfx = "" if dataset == "goose" else f"{dataset}_"; ssfx = "" if seed == 0 else f"_s{seed}"
    fn = f"{ROOT}/results/m2f_predclass_{pfx}{split}{ssfx}.npz"
    np.savez_compressed(fn, predclass=np.stack(preds), MSP_M2F=np.stack(msps), keys=np.stack(keys))
    print(f"[saved] {fn} (NC={NC}, frames{N})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose")
    ap.add_argument("--split", default="test", choices=["test", "val"]); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); main(a.dataset, a.split, a.seed)
