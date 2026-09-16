#!/usr/bin/env python
"""
#5 MSP/MaxLogit/Energy evaluation for SegFormer (per-pixel), compared against Mask2Former (mask) / frozen DINOv2 linear head.
Usage (GPU): python segformer_eval.py --dataset goose --split test
"""
import os, time, argparse, json, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import IGNORE, ROOT
from datasets import get_dataset
from transformers import SegformerForSemanticSegmentation
from segformer_train import CKPT, make_proc, ckpt_path, logits_full
import metrics as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load_short(ds):
    try: return json.load(open(f"{ROOT}/results/segformer_{ds}_meta.json"))["short"]
    except Exception: return 512

@torch.no_grad()
def score_maps(model, proc, img):
    lo = logits_full(model, proc, img)          # (NC,H,W)
    msp = 1 - F.softmax(lo, 0).max(0).values
    maxl = -lo.max(0).values
    en = -torch.logsumexp(lo, 0)
    return {k: v.cpu().numpy() for k, v in [("MSP", msp), ("MaxLogit", maxl), ("Energy", en)]}

def main(dataset="goose", split="test"):
    DS, ol = get_dataset(dataset); onto = ol(); NC = onto["num_train"]
    proc = make_proc(load_short(dataset))
    model = SegformerForSemanticSegmentation.from_pretrained(CKPT, num_labels=NC, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(ckpt_path(dataset), map_location="cpu")); model.to(DEV).eval()
    dst = DS(split, mode="test"); N = len(dst)
    methods = ["MSP", "MaxLogit", "Energy"]; buf = {k: [] for k in methods}; gts = []; keys = []; t0 = time.time()
    for i in range(N):
        s = dst[i]; sc = score_maps(model, proc, s["image"])
        for k in methods: buf[k].append(sc[k][::4, ::4].astype(np.float16))
        gts.append(s["target"][::4, ::4]); keys.append(s["label_key"][::4, ::4].copy())
        if i % 200 == 0: print(f"  {split} {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    out = {}
    print(f"\n=== SegFormer per-pixel ({split}) ===")
    print(f"{'method':10s} {'AP':>7s} {'AUROC':>7s} {'FPR95':>7s}")
    for k in methods:
        r = M.pixel_pr_auroc_fpr95(buf[k], gts); out[k] = r
        print(f"{k:10s} {r['AP']:7.4f} {r['AUROC']:7.4f} {r['FPR95']:7.4f}")
    pfx = "" if dataset == "goose" else f"{dataset}_"
    json.dump(out, open(f"{ROOT}/results/segformer_{pfx}{split}.json", "w"), indent=2)
    np.savez_compressed(f"{ROOT}/results/segformer_scores_{pfx}{split}.npz",
                        keys=np.stack(keys), **{k: np.stack(buf[k]) for k in methods})
    print(f"[saved] segformer_{pfx}{split}.json (+scores)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose", choices=["goose","rellis","wildscenes","rugd"])
    ap.add_argument("--split", default="test", choices=["test","val"]); a = ap.parse_args()
    main(a.dataset, a.split)
