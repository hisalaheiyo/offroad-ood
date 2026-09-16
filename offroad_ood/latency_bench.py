#!/usr/bin/env python
"""
#2 Cost transparency: per-method-family inference latency (ms/frame)+FPS+params, for the cost column of Table 3 (relevant to robotics deployment).
Timed on a fixed N frames of GOOSE test (warmup+cuda.synchronize for accuracy). Methods: DINOv2-kNN / SegFormer / Mask2Former(RbA).
Usage (GPU): python latency_bench.py --n 50
"""
import os, time, argparse, json, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import ROOT
from datasets import get_dataset
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def nparams(m): return sum(p.numel() for p in m.parameters())

def time_method(fn, imgs, warmup=3):
    for i in range(warmup): fn(imgs[i % len(imgs)])
    sync(); t0 = time.time()
    for im in imgs: fn(im)
    sync(); return (time.time() - t0) / len(imgs) * 1000   # ms/frame

def main(n=50):
    DS, ol = get_dataset("goose"); onto = ol(); NC = onto["num_train"]
    ds = DS("test", mode="test"); imgs = [ds[i]["image"] for i in range(0, min(n, len(ds)))]
    out = {}

    # 1. DINOv2-kNN
    from dinov2_baseline import load_model, extract_patches, bank_path
    m, mean, std = load_model(); bank = torch.load(bank_path("goose")).to(DEV); bT = bank.T
    @torch.no_grad()
    def knn(img):
        f, (gh, gw) = extract_patches(m, mean, std, img)
        sc = torch.empty(f.shape[0], device=DEV)
        for s in range(0, f.shape[0], 2048): sc[s:s+2048] = 1 - (f[s:s+2048] @ bT).max(1).values
        return F.interpolate(sc.reshape(gh, gw)[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
    out["DINOv2-kNN"] = dict(ms=time_method(knn, imgs), params=nparams(m),
                             bank_MB=bank.numel() * 4 / 1e6, note="frozen ViT-S (not trained); bank=storage, not params")
    del m, bank; torch.cuda.empty_cache()

    # 2. SegFormer
    from transformers import SegformerForSemanticSegmentation
    from segformer_train import CKPT as SCKPT, make_proc as smake, ckpt_path as sckpt, logits_full
    from segformer_eval import load_short as sshort
    sm = SegformerForSemanticSegmentation.from_pretrained(SCKPT, num_labels=NC, ignore_mismatched_sizes=True)
    sm.load_state_dict(torch.load(sckpt("goose"), map_location="cpu")); sm.to(DEV).eval(); sproc = smake(sshort("goose"))
    @torch.no_grad()
    def segf(img):
        lo = logits_full(sm, sproc, img); return 1 - F.softmax(lo, 0).max(0).values
    out["SegFormer-b2"] = dict(ms=time_method(segf, imgs), params=nparams(sm), note="per-pixel transformer")
    del sm; torch.cuda.empty_cache()

    # 3. Mask2Former (RbA)
    from transformers import Mask2FormerForUniversalSegmentation
    from m2f_train import CKPT as MCKPT, make_proc as mmake, ckpt_path as mckpt, load_short as mshort
    from m2f_rba_eval import rba_maps
    mm = Mask2FormerForUniversalSegmentation.from_pretrained(MCKPT, num_labels=NC, ignore_mismatched_sizes=True)
    mm.load_state_dict(torch.load(mckpt("goose"), map_location="cpu")); mm.to(DEV).eval(); mproc = mmake(mshort("goose"))
    @torch.no_grad()
    def m2f(img): return rba_maps(mm, mproc, img, NC)[0]
    out["Mask2Former(RbA/MSP-M2F)"] = dict(ms=time_method(m2f, imgs), params=nparams(mm), note="mask-transformer; OURS=+kNN")

    # OURS = Mask2Former + kNN (sum of the two run serially; method=M2F score + cDNP terrain-gate)
    out["OURS(M2F+kNN)"] = dict(ms=out["Mask2Former(RbA/MSP-M2F)"]["ms"] + out["DINOv2-kNN"]["ms"],
                                params=out["Mask2Former(RbA/MSP-M2F)"]["params"] + out["DINOv2-kNN"]["params"],
                                note="M2F + cDNP terrain-gate (serial)")
    print(f"\n=== inference cost (GOOSE, {len(imgs)} frames, {DEV}) ===")
    print(f"{'method':28s} {'ms/frame':>9s} {'FPS':>6s} {'params(M)':>10s}")
    for k, v in out.items():
        print(f"{k:28s} {v['ms']:9.1f} {1000/v['ms']:6.1f} {v['params']/1e6:10.1f}  {v['note']}")
    json.dump(out, open(f"{ROOT}/results/latency_bench.json", "w"), indent=2)
    print("[saved] latency_bench.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); a = ap.parse_args(); main(a.n)
