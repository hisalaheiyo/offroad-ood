#!/usr/bin/env python
"""
Confusion-matrix validation (A4): verify that held-out OOD classes are indeed "anomalous" (the ID model assigns them low confidence / confuses them) + adjudicate borderline classes (rock/asphalt).
Uses the trained seghead on test: per pixel take argmax(ID prediction) + max-softmax (confidence). Aggregate by ground-truth label_key.
- OOD classes: mean confidence should be low + predictions dispersed => truly anomalous (benchmark is valid).
- borderline (rock/asphalt, currently assigned to ID): if confidently and correctly predicted => keep as ID; if confused => reconsider.
Usage (GPU): python confusion_matrix.py
"""
import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ontology import ROOT
from dinov2_baseline import load_model, DEV
from train_seghead import raw_patch_feats, head_path
from datasets import get_dataset

@torch.no_grad()
def main(dataset="goose", coarse=None):
    DS,onto_loader=get_dataset(dataset); o=onto_loader(); info=o["info"]; NC=o["num_train"]
    name=lambda k:info[k]["name"]; role=lambda k:info[k]["eff_role"]
    if coarse:
        assert dataset in ("goose","rellis"),"granularity ablation supports goose/rellis"
        from coarse_groups import coarse_lut
        _,NC,gn=coarse_lut(o,coarse,dataset); trainid2name={i:gn[i] for i in range(NC)}
        print(f"[coarse {coarse}/{dataset}] NC={NC}",flush=True)
    else:
        trainid2name={o["key2trainid"][k]:name(k) for k in o["id_keys"]}
    m,mean,std=load_model(); head=nn.Linear(384,NC).to(DEV); head.load_state_dict(torch.load(head_path(dataset,coarse))); head.eval()
    ds=DS("test",mode="test"); N=len(ds)
    conf_sum=np.zeros(64); cnt=np.zeros(64); predhist=np.zeros((64,NC)); t0=time.time()
    for i in range(N):
        s=ds[i]; f,(gh,gw)=raw_patch_feats(m,mean,std,s["image"])
        lo=head(f.reshape(-1,384)).reshape(gh,gw,-1).permute(2,0,1)
        up=F.interpolate(lo[None],size=s["image"].shape[:2],mode="bilinear",align_corners=False)[0]
        prob=F.softmax(up,0); conf,pred=prob.max(0)
        conf=conf.cpu().numpy().ravel(); pred=pred.cpu().numpy().ravel(); key=s["label_key"].ravel()
        for k in np.unique(key):
            mk=key==k; conf_sum[k]+=conf[mk].sum(); cnt[k]+=mk.sum()
            ph=np.bincount(pred[mk],minlength=NC); predhist[k]+=ph
        if i%400==0: print(f"  {i}/{N} ({time.time()-t0:.0f}s)",flush=True)
    mean_conf=conf_sum/np.maximum(cnt,1)
    # Overall mean confidence over ID classes (reference baseline)
    id_conf=conf_sum[o["id_keys"]].sum()/max(cnt[o["id_keys"]].sum(),1)
    print(f"\n=== Overall mean confidence over ID classes (baseline) = {id_conf:.3f} ===")
    print("\n=== OOD classes: mean confidence (should be < ID) + top-3 predicted ID classes (should be dispersed / not the class itself) ===")
    out={"id_baseline_conf":float(id_conf),"ood":{},"borderline":{}}
    for k in o["ood_keys"]:
        if cnt[k]<100: continue
        top=np.argsort(-predhist[k])[:3]
        tt=", ".join(f"{trainid2name[t]}{int(predhist[k][t]/cnt[k]*100)}%" for t in top)
        print(f"  {name(k):16s} conf={mean_conf[k]:.3f}  confused toward: {tt}")
        out["ood"][name(k)]=dict(conf=float(mean_conf[k]),n=int(cnt[k]))
    # Overall mean confidence over OOD classes (key for the granularity ablation: coarser granularity should raise it)
    ood_conf=conf_sum[o["ood_keys"]].sum()/max(cnt[o["ood_keys"]].sum(),1)
    out["ood_baseline_conf"]=float(ood_conf)
    print(f"\n=== Overall mean confidence over OOD classes = {ood_conf:.3f} (ID baseline={id_conf:.3f}); mechanism: coarser granularity -> higher OOD confidence -> logit fails ===")
    if not coarse:
        print("\n=== borderline (rock/asphalt, currently assigned to ID): are they confidently and correctly predicted? ===")
        for k in o["id_keys"]:
            if info[k]["is_border"] and cnt[k]>=100:
                tid=o["key2trainid"][k]; selfpred=predhist[k][tid]/cnt[k]
                print(f"  {name(k):16s} conf={mean_conf[k]:.3f} self-prediction rate={selfpred*100:.0f}% {'✅keep as ID' if selfpred>0.5 else '⚠️confused, reconsider'}")
                out["borderline"][name(k)]=dict(conf=float(mean_conf[k]),self_pred=float(selfpred))
    fn=("confusion_matrix.json" if dataset=="goose" else f"confusion_matrix_{dataset}.json") if not coarse else f"confusion_matrix_coarse_{'' if dataset=='goose' else dataset+'_'}{coarse}.json"
    json.dump(out,open(f"{ROOT}/results/{fn}","w"),indent=2)
    print(f"[saved] results/{fn}")

if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--dataset",default="goose",choices=["goose","wildscenes","rellis","rugd","rellis_comp","rugd_comp"])
    ap.add_argument("--coarse",default=None,choices=["L11","L7","L4"]); a=ap.parse_args()
    main(a.dataset, coarse=a.coarse)
