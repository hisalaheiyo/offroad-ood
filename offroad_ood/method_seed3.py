#!/usr/bin/env python
"""
GOOSE 3-seed characterization: AP/FPR95 mean±std of RbA/MSP-M2F/OURS across seed{0,1,2}.
Confirms the finding that "GOOSE FPR95 is seed-unstable (AP stable)". Reuses method_seed.metrics_seed.
Usage (CPU): python method_seed3.py --dataset goose --seeds 0,1,2
"""
import argparse, json, numpy as np
from ontology import ROOT
from method_seed import metrics_seed
R = f"{ROOT}/results"

def main(dataset="goose", seeds=(0, 1, 2)):
    rs = {s: metrics_seed(dataset, s) for s in seeds}
    print(f"\n=== {dataset} {len(seeds)}-seed (seeds={list(seeds)}) ===")
    print(f"{'method':10s} {'AP mean±std':>18s} {'FPR95 mean±std':>20s}  {'AP CV':>6s} {'FPR CV':>7s}")
    out = {}
    for m in ["RbA", "MSP_M2F", "OURS"]:
        aps = np.array([rs[s][m]["AP"] for s in seeds]); fps = np.array([rs[s][m]["FPR95"] for s in seeds])
        out[m] = dict(AP_mean=float(aps.mean()), AP_std=float(aps.std()), FPR_mean=float(fps.mean()), FPR_std=float(fps.std()),
                      AP_vals=aps.tolist(), FPR_vals=fps.tolist())
        cvap = aps.std() / (aps.mean() + 1e-9); cvfp = fps.std() / (fps.mean() + 1e-9)
        print(f"{m:10s} {aps.mean():7.3f}±{aps.std():.3f}      {fps.mean():7.3f}±{fps.std():.3f}       {cvap*100:5.1f}% {cvfp*100:6.1f}%")
    betas = [rs[s]["beta"] for s in seeds]
    print(f"  β per seed: {betas}")
    print(f"\nConfirmation: AP's CV (coefficient of variation) should be << FPR95's CV -> AP stable / FPR95 unstable. (CV=std/mean)")
    json.dump({str(s): rs[s] for s in seeds} | {"summary": out}, open(f"{R}/method_seed3_{'' if dataset=='goose' else dataset+'_'}test.json", "w"), indent=2)
    print("[saved]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="goose"); ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(); main(a.dataset, tuple(int(x) for x in a.seeds.split(",")))
