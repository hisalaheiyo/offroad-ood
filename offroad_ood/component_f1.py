#!/usr/bin/env python
"""
component-level F1 (SMIYC-style connected components + sIoU, averaged over τ∈{0.25..0.75}). Robotics alignment: measures "whether the whole small object is detected".
Correctness: the binarization threshold is selected on val (TPR95 operating point), applied to test. The threshold is never selected on test.
Methods: seghead (MSP/MaxLogit/Energy/SML), cDNP, Mahalanobis (those with val+test dumps).
Usage (CPU node): python component_f1.py
"""
import json, numpy as np
from ontology import load_goose_ontology, IGNORE, ROOT
import metrics as M

def to_lists(S, keys, test_lut):
    gt=test_lut[keys]
    return [S[i] for i in range(S.shape[0])], [gt[i] for i in range(S.shape[0])]

def best_f1_thr(scores, gts):
    """Sweep candidate thresholds on val, select the one that maximizes component-F1 (standard operating-point selection). TPR95 is wrong (false positives swamp it)."""
    s=np.concatenate([sc[g!=255].ravel() for sc,g in zip(scores,gts)])
    g=np.concatenate([g[g!=255].ravel() for g in gts]); ood=s[g==1]
    cands=np.percentile(ood, np.arange(40,100,8))           # 40..96 quantile (from balanced to high recall)
    best_t,best_f=cands[0],-1
    for t in np.unique(cands):
        f=M.component_f1_aggregate(scores,gts,thr=float(t))["component_F1"]
        if f>best_f: best_f,best_t=f,float(t)
    return best_t, best_f

def main(dataset="goose"):
    from datasets import get_dataset
    R=ROOT+"/results"; pfx="" if dataset=="goose" else f"{dataset}_"
    _,onto_loader=get_dataset(dataset); o=onto_loader(); test_lut=np.full(256,IGNORE,np.uint8)
    for k,v in o["test_lut"].items(): test_lut[k]=v
    sh=f"seghead_scores_{pfx}"; cd=f"cdnp_scores_{pfx}"; mh=f"mahalanobis_scores_{pfx}"
    M7={"MSP":(f"{sh}val.npz",f"{sh}test.npz","MSP"),"MaxLogit":(f"{sh}val.npz",f"{sh}test.npz","MaxLogit"),
        "Energy":(f"{sh}val.npz",f"{sh}test.npz","Energy"),"SML":(f"{sh}val.npz",f"{sh}test.npz","SML"),
        "cDNP_k5":(f"{cd}val.npz",f"{cd}test.npz","scores"),"Mahalanobis":(f"{mh}val.npz",f"{mh}test.npz","scores")}
    # shared keys (all dumps consistent in same split, same order, same downsample): test from dinov2, val from seghead (both contain keys)
    Ktest=np.load(f"{R}/dinov2_scores_{pfx}test.npz")["keys"]
    Kval =np.load(f"{R}/{sh}val.npz")["keys"]
    out={}
    print(f"{'method':12s} {'τ(val,bestF1)':>13s} {'component-F1(test)':>18s}")
    for name,(vf,tf,key) in M7.items():
        try:
            Sv=np.load(f"{R}/{vf}")[key].astype(np.float32); St=np.load(f"{R}/{tf}")[key].astype(np.float32)
        except FileNotFoundError as e:
            print(f"{name:12s}  skip (missing {e.filename.split('/')[-1]})"); continue
        assert Sv.shape==Kval.shape and St.shape==Ktest.shape, f"{name} shape misaligned"
        vs,vg=to_lists(Sv,Kval,test_lut); ts,tg=to_lists(St,Ktest,test_lut)
        print(f"  [{name}] val threshold sweep to select best-F1...",flush=True)
        thr,valf1=best_f1_thr(vs,vg)                           # select the F1-maximizing threshold on val
        cf=M.component_f1_aggregate(ts,tg,thr=thr)             # compute on test
        out[name]=dict(thr=thr,val_F1=valf1,component_F1=cf["component_F1"],per_tau=cf["per_tau"])
        print(f"{name:12s} {thr:13.4f} {cf['component_F1']:18.4f}  (valF1={valf1:.3f})")
    fn="component_f1.json" if dataset=="goose" else f"component_f1_{dataset}.json"
    json.dump(out,open(f"{R}/{fn}","w"),indent=2)
    print(f"[saved] results/{fn}")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp"]); a=ap.parse_args()
    main(a.dataset)
