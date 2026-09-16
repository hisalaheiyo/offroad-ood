#!/usr/bin/env python
"""
S2: RbA (Nayal et al. ICCV23, "Rejected by All") modern mask-transformer OOD baseline.
Evaluated on the fine-tuned Mask2Former, compared against classic per-pixel scores (kNN/MSP/...) on the same test split / same metric.
RbA scoring (faithful to the original paper):
  probs_q,c = softmax(class_logits_q)[:C]           # drop no-object
  mask_q(x) = sigmoid(mask_logits_q(x))
  L_c(x)    = Σ_q probs_q,c · mask_q(x)              # standard Mask2Former semantic logit
  RbA(x)    = -Σ_c L_c(x)                            # "rejected by all known classes" -> if no class claims the pixel, anomaly is high
Correctness reflection:
  (1) Compute L at the model output resolution, then interpolate the RbA map to full resolution, [::4,::4] aligned with GT (same as other methods).
  (2) test mode target=binary OOD (0/1/255); keys=label_key for per-class decomposition.
  (3) Add MSP-on-M2F control (-max_c softmax semantic) to show the conclusion is not specific to RbA.
Usage (GPU): python m2f_rba_eval.py --dataset goose
"""
import os, time, argparse, numpy as np, torch, torch.nn.functional as F, json
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import ROOT
from datasets import get_dataset
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from m2f_train import CKPT, make_proc, ckpt_path, SHORT, load_short
import metrics as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def rba_maps(model, proc, img, NC):
    """Returns (rba_map HxW, msp_map HxW) numpy, full resolution."""
    H, W = img.shape[:2]
    enc = proc(images=[img], return_tensors="pt"); px = enc["pixel_values"].to(DEV)
    out = model(pixel_values=px)
    cls = out.class_queries_logits[0]            # (Q, C+1)
    msk = out.masks_queries_logits[0]            # (Q, h, w)
    probs = cls.softmax(-1)[:, :NC]              # (Q, C) drop no-object
    mprob = msk.sigmoid()                        # (Q, h, w)
    L = torch.einsum("qc,qhw->chw", probs, mprob)   # (C, h, w) semantic score
    rba = -L.sum(0)                              # (h, w) RbA anomaly
    msp = -L.max(0).values                       # (h, w) MSP-style (lower max-class score means more anomalous)
    def up(m):
        return F.interpolate(m[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return up(rba), up(msp)

def main(dataset="goose", split="test", seed=0):
    DS, onto_loader = get_dataset(dataset); onto = onto_loader(); NC = onto["num_train"]
    proc = make_proc(load_short(dataset))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT, num_labels=NC, ignore_mismatched_sizes=True)
    sd = torch.load(ckpt_path(dataset, seed), map_location="cpu"); model.load_state_dict(sd); model.to(DEV).eval()
    print(f"[rba] {dataset} split={split} seed={seed} NC={NC} loading {ckpt_path(dataset, seed)}", flush=True)
    dst = DS(split, mode="test"); N = len(dst)
    buf_r, buf_m, gts, keys = [], [], [], []; t0 = time.time()
    for i in range(N):
        s = dst[i]; rba, msp = rba_maps(model, proc, s["image"], NC)
        buf_r.append(rba[::4, ::4].astype(np.float16)); buf_m.append(msp[::4, ::4].astype(np.float16))
        gts.append(s["target"][::4, ::4]); keys.append(s["label_key"][::4, ::4].copy())
        if i % 200 == 0: print(f"  {i}/{N} ({time.time()-t0:.0f}s)", flush=True)
    out = {}
    print(f"\n=== RbA / MSP-M2F ({split}) ===")
    print(f"{'method':12s} {'AP':>7s} {'AUROC':>7s} {'FPR95':>7s}")
    for name, buf in [("RbA", buf_r), ("MSP-M2F", buf_m)]:
        r = M.pixel_pr_auroc_fpr95(buf, gts); out[name] = r
        print(f"{name:12s} {r['AP']:7.4f} {r['AUROC']:7.4f} {r['FPR95']:7.4f}")
    pfx = "" if dataset == "goose" else f"{dataset}_"
    ssfx = "" if seed == 0 else f"_s{seed}"                              # seed suffix so seed0 is not overwritten
    json.dump(out, open(f"{ROOT}/results/m2f_rba_{pfx}{split}{ssfx}.json", "w"), indent=2)
    np.savez_compressed(f"{ROOT}/results/m2f_rba_scores_{pfx}{split}{ssfx}.npz",
                        RbA=np.stack(buf_r), MSP_M2F=np.stack(buf_m), keys=np.stack(keys))
    print(f"[saved] m2f_rba_{pfx}{split}{ssfx}.json (+scores npz)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose", choices=["goose", "rellis", "wildscenes", "rugd", "rugd_comp"])
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); main(a.dataset, a.split, a.seed)
