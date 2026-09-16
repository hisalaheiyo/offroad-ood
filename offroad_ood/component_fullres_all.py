#!/usr/bin/env python
"""
Point 3: full-resolution mean component-F1 for ALL object-level methods
(per-pixel kNN/cDNP/MaxLogit, mask RbA/MSP-M2F, and Ours-Score = fixed beta=1 fusion).

Motivation: at 1/4 resolution the median GOOSE anomaly component is ~4 px and min_pix=10
filters ~66% of GT components -> object-level metric is a resolution artifact. Here we
score each TEST frame at NATIVE resolution, extract connected components on native GT,
and report the threshold-averaged component-F1 (renamed "mean component F1").

Metric (matches the quarter-res F1* definition, but native res):
  - for each method, 9 binarization thresholds = percentiles {5..70} (TPR95->TPR30) of the
    method's TEST OOD-pixel scores (derived from cached 1/4-res dumps; native ~= 1/4 in dist.)
  - at each threshold, component-F1 averaged over sIoU/PPV taus {0.25..0.75} (11 values)
  - mean over the 9 thresholds -> reported number. min_pix=10 at NATIVE resolution.

Writes results/component_fullres_all_<ds>.json only. Does NOT touch existing results.
GPU. Run per dataset:  python component_fullres_all.py --dataset goose
"""
import os, json, time, argparse, gc, numpy as np, torch, torch.nn as nn
os.environ.setdefault("HF_HOME", "./.hf_cache")
from ontology import ROOT, IGNORE
from datasets import get_dataset
from dinov2_baseline import load_model, bank_path, DEV
from cdnp_eval import score_knn
from eval_seghead import frame_logits, scores_from_logits
from train_seghead import head_path
from transformers import Mask2FormerForUniversalSegmentation
from m2f_train import CKPT, make_proc, ckpt_path, load_short
from m2f_rba_eval import rba_maps
import metrics as M

TAUS = np.arange(0.25, 0.751, 0.05)          # 11 sIoU/PPV thresholds
PCTS = np.linspace(5, 70, 9)                  # 9 binarization thresholds (TPR95..TPR30)
METHODS = ["kNN", "cDNP", "MaxLogit", "RbA", "MSP-M2F", "Ours-Score"]
MINPIX = 10                                    # min component size at NATIVE res (set via --minpix)
R = f"{ROOT}/results"


def zstat_valid(fn, key, tl):
    """val-ID mean/std of a cached 1/4-res score (ID = test_lut label 0)."""
    z = np.load(f"{R}/{fn}"); s = z[key].astype(np.float32); idv = tl[z["keys"]] == 0
    return float(s[idv].mean()), float(s[idv].std() + 1e-6)


def cached_test_ood_scores(dataset, tl):
    """per-method TEST OOD-pixel scores from cached 1/4-res dumps -> for threshold percentiles."""
    pfx = "" if dataset == "goose" else f"{dataset}_"

    def load(fn, key):
        z = np.load(f"{R}/{fn}"); return z[key].astype(np.float32), tl[z["keys"]]
    kn, gk_kn = load(f"dinov2_scores_{pfx}test.npz", "scores")
    cd, gk_cd = load(f"cdnp_scores_{pfx}test.npz", "scores")
    sg = np.load(f"{R}/seghead_scores_{pfx}test.npz"); ml = sg["MaxLogit"].astype(np.float32)
    assert ml.shape == kn.shape, "seghead vs dinov2 shape mismatch (MaxLogit key-reuse invalid)"
    gk_ml = gk_kn                                            # seghead dump lacks 'keys'; same frames/[::4,::4] as dinov2
    rb = np.load(f"{R}/m2f_rba_scores_{pfx}test.npz"); rba = rb["RbA"].astype(np.float32); mspm = rb["MSP_M2F"].astype(np.float32); gk_rb = tl[rb["keys"]]
    # z-stats for fused (val-ID)
    muM, sdM = zstat_valid(f"m2f_rba_scores_{pfx}val.npz", "MSP_M2F", tl)
    muC, sdC = zstat_valid(f"cdnp_scores_{pfx}val.npz", "scores", tl)
    assert mspm.shape == cd.shape and np.array_equal(rb["keys"], np.load(f"{R}/cdnp_scores_{pfx}test.npz")["keys"]), "MSP-M2F vs cDNP test keys misaligned (fused invalid)"
    fused = (mspm - muM) / sdM + (cd - muC) / sdC            # beta=1, aligned with cdnp/rba test keys
    ood = {"kNN": kn[gk_kn == 1], "cDNP": cd[gk_cd == 1], "MaxLogit": ml[gk_ml == 1],
           "RbA": rba[gk_rb == 1], "MSP-M2F": mspm[gk_rb == 1], "Ours-Score": fused[gk_rb == 1]}
    thr = {m: np.percentile(v, PCTS).tolist() for m, v in ood.items()}
    return thr, (muM, sdM, muC, sdC)


@torch.no_grad()
def native_scores(m, mean, std, bank, head, m2f, proc, NC, zst, img):
    """all 6 methods, native (H,W), for one frame."""
    out = {}
    kn = score_knn(m, mean, std, bank, img, ks=(1, 5), B=512)  # B=512: sim matrix ~2GB -> fits 16G GPU
    out["kNN"] = kn[1]; cdnp = kn[5]; out["cDNP"] = cdnp
    lo = frame_logits(m, mean, std, head, img)               # (C,H,W)
    out["MaxLogit"] = scores_from_logits(lo)["MaxLogit"].cpu().numpy()
    del lo
    rba, mspm = rba_maps(m2f, proc, img, NC)                  # native RbA, MSP-M2F
    out["RbA"] = rba; out["MSP-M2F"] = mspm
    muM, sdM, muC, sdC = zst
    out["Ours-Score"] = (mspm - muM) / sdM + (cdnp - muC) / sdC   # beta=1 fused, native
    return out


def main(dataset="goose", limit=0):
    t0 = time.time()
    DS, ol = get_dataset(dataset); onto = ol(); NC = onto["num_train"]
    tl = np.full(256, IGNORE, np.uint8)
    for k, v in onto["test_lut"].items(): tl[k] = v

    thr, zst = cached_test_ood_scores(dataset, tl)
    print(f"[{time.time()-t0:.0f}s] thresholds ready: " + " ".join(f"{k}:{len(v)}" for k, v in thr.items()), flush=True)

    # models
    m, mean, std = load_model()
    bank = torch.load(bank_path(dataset)).to(DEV)
    head = nn.Linear(384, NC).to(DEV); head.load_state_dict(torch.load(head_path(dataset))); head.eval()
    proc = make_proc(load_short(dataset))
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT, num_labels=NC, ignore_mismatched_sizes=True)
    m2f.load_state_dict(torch.load(ckpt_path(dataset, 0), map_location="cpu")); m2f.to(DEV).eval()
    print(f"[{time.time()-t0:.0f}s] models loaded", flush=True)

    # accumulate tp/fn/fp per method x threshold-idx x tau
    acc = {mth: [{float(t): dict(tp=0, fn=0, fp=0) for t in TAUS} for _ in range(len(PCTS))] for mth in METHODS}
    dst = DS("test", mode="test"); N = len(dst)
    if limit: N = min(N, limit)
    print(f"[{time.time()-t0:.0f}s] native test pass ({N} frames{' SMOKE' if limit else ''})...", flush=True)
    for i in range(N):
        s = dst[i]; g = tl[s["label_key"]]                  # native GT 0/1/255
        sc = native_scores(m, mean, std, bank, head, m2f, proc, NC, zst, s["image"])
        for mth in METHODS:
            for ti, thv in enumerate(thr[mth]):
                r = M.component_f1_single(sc[mth], g, float(thv), TAUS, min_pix=MINPIX)
                for t in TAUS:
                    for kk in ("tp", "fn", "fp"):
                        acc[mth][ti][float(t)][kk] += r[float(t)][kk]
        del sc
        if i % 100 == 0:
            print(f"  {i}/{N} ({time.time()-t0:.0f}s)", flush=True); gc.collect()

    out = {}
    print(f"\n{'method':12s} {'mean component F1 (native)':>26s}")
    for mth in METHODS:
        per_thr = []
        for ti in range(len(PCTS)):
            f1s = [acc[mth][ti][float(t)]["tp"] / max(acc[mth][ti][float(t)]["tp"] + 0.5 * (acc[mth][ti][float(t)]["fn"] + acc[mth][ti][float(t)]["fp"]), 1e-9) for t in TAUS]
            per_thr.append(float(np.mean(f1s)))
        val = float(np.mean(per_thr))
        out[mth] = dict(mean_component_F1=val, per_threshold=per_thr)
        print(f"{mth:12s} {val:26.4f}")
    fn = f"component_fullres_all_{dataset}_mp{MINPIX}{'_n' + str(limit) if limit else ''}.json"
    json.dump(out, open(f"{R}/{fn}", "w"), indent=2)
    print(f"[saved] results/{fn} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="goose", choices=["goose", "rellis", "wildscenes", "rugd"])
    ap.add_argument("--limit", type=int, default=0, help="cap #test frames (smoke test)")
    ap.add_argument("--minpix", type=int, default=10, help="min component size at native resolution")
    a = ap.parse_args()
    MINPIX = a.minpix
    main(a.dataset, a.limit)
