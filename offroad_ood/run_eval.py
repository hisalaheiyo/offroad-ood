#!/usr/bin/env python
"""
OffRoad-OOD — evaluation pipeline runner (method-agnostic)
Given a score_fn(image_rgb)->anomaly_score(HxW float), evaluates on the specified split, produces a table with CIs, saves json.
Usage:
  python run_eval.py --method trivial_nongreen --split test --limit 100     # pipeline validation (CPU)
  python run_eval.py --method dinov2_knn --split test                        # real baseline (GPU)
"""
import argparse, json, time, numpy as np
from goose_dataset import GooseOOD
import metrics as M

# ---------- baselines registry ----------
def trivial_nongreen(img):
    """Weak heuristic: the less 'green', the more anomalous. Vegetation (green)=low score, man-made=high score. Pure CPU, for pipeline validation."""
    r,g,b = img[...,0].astype(float), img[...,1].astype(float), img[...,2].astype(float)
    greenness = g - 0.5*(r+b)
    return -greenness   # non-green -> high anomaly score

def random_score(img):
    rng=np.random.default_rng(abs(hash(img.tobytes()))%(2**32))
    return rng.normal(0,1,img.shape[:2])

_DINOV2=None
def dinov2_knn(img):
    """Nearest-neighbor cosine distance between frozen DINOv2 patch features and the ID memory bank as anomaly. Requires GPU; bank pre-built."""
    raise NotImplementedError("dinov2_knn is injected by a GPU script after build_bank; see dinov2_baseline.py")

METHODS={"trivial_nongreen":trivial_nongreen, "random":random_score}

def evaluate(method, split, limit=None, n_boot=0, downsample=4, do_component=False):
    ds = GooseOOD(split, mode="test")   # test mode: GT=1/0/255
    fn = METHODS[method]
    N = len(ds) if limit is None else min(limit, len(ds))
    scores=[]; gts=[]
    t0=time.time()
    for i in range(N):
        s = ds[i]
        sc = fn(s["image"]).astype(np.float32)
        gt = s["target"]
        if downsample>1:   # downsample to save memory/speed up (pixel metrics are insensitive)
            sc=sc[::downsample,::downsample]; gt=gt[::downsample,::downsample]
        scores.append(sc); gts.append(gt)
    t_load=time.time()-t0
    # pixel-level (concatenate all frames)
    pix = M.pixel_pr_auroc_fpr95(scores, gts)
    out=dict(method=method, split=split, n_frames=N, sec_score=round(t_load,1),
             pixel=pix, downsample=downsample)
    # component-level (optional; heavy, off by default during pipeline validation)
    if do_component:
        allsc=np.concatenate([s.ravel() for s in scores])
        thr=float(np.median(allsc)+np.std(allsc))
        out["component"]=M.component_f1_aggregate(scores, gts, thr=thr); out["thr"]=thr
    # bootstrap CI on AP (optional; n_boot=0 disables)
    if n_boot>0:
        out["AP_CI"]=M.bootstrap_ci(scores, gts, lambda s,g:M.pixel_pr_auroc_fpr95(s,g)['AP'], n_boot=n_boot)
    out["sec_total"]=round(time.time()-t0,1)
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--method", default="trivial_nongreen")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n_boot", type=int, default=0)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--component", action="store_true")
    a=ap.parse_args()
    res=evaluate(a.method, a.split, a.limit, a.n_boot, a.downsample, a.component)
    print("="*60); print(f"method={res['method']} split={res['split']} N={res['n_frames']} ({res['sec_total']}s, load+score {res['sec_score']}s)")
    print("="*60)
    print(f"  pixel  AP={res['pixel']['AP']:.4f}  AUROC={res['pixel']['AUROC']:.4f}  FPR95={res['pixel']['FPR95']:.4f}  (n_pos={res['pixel']['n_pos']})")
    if "AP_CI" in res: print(f"  AP 95%CI=[{res['AP_CI']['lo']:.4f}, {res['AP_CI']['hi']:.4f}]  mean={res['AP_CI']['mean']:.4f}")
    if "component" in res: print(f"  component-F1={res['component']['component_F1']:.4f}  (thr={res['thr']:.3f})")
    import os; os.makedirs(f"{M.__file__.rsplit('/',1)[0]}/../results", exist_ok=True)
    p=f"{M.__file__.rsplit('/',1)[0]}/../results/{a.method}_{a.split}.json"
    json.dump(res, open(p,"w"), indent=2)
    print(f"  [saved] {p}")
