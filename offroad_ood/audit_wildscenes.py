#!/usr/bin/env python
"""WildScenes feasibility check: number of anomaly frames per split + per-OOD class long-tail + ignore fraction."""
import sys, numpy as np
from collections import Counter
from wildscenes_dataset import WildScenesOOD
from ontology import load_wildscenes_ontology
MIN_OOD_PX=10
def main():
    o=load_wildscenes_ontology(); info=o["info"]
    name=lambda k:info[k]["name"]; tier=lambda k:info[k]["tier"]
    ood_keys=set(o["ood_keys"]); ign_keys=set(o["ignore_keys"])
    for sp in ("train","val","test"):
        d=WildScenesOOD(sp,mode="test"); N=len(d)
        cls_fr=Counter(); tier_fr=Counter(); n_anom=n_pure=0; ign_fr=[]
        for i in range(N):
            k=d[i]["label_key"]; bc=np.bincount(k.ravel(),minlength=256)
            tot=k.size; ign=sum(int(bc[j]) for j in ign_keys)+int(bc[255]); ign_fr.append(ign/tot)
            tiers=set(); has=False
            for j in ood_keys:
                if bc[j]>=MIN_OOD_PX: cls_fr[name(j)]+=1; tiers.add(tier(j)); has=True
            for t in tiers: tier_fr[t]+=1
            if has: n_anom+=1
            else:
                oodpx=sum(int(bc[j]) for j in ood_keys)
                if oodpx==0: n_pure+=1
            if i%800==0: print(f"  {sp} {i}/{N}",flush=True)
        print(f"\n[{sp}] {N} frames | anomaly{n_anom} {'✅≥100' if n_anom>=100 else '🔴<100'} | pure ID{n_pure} | ignore P50={np.percentile(ign_fr,50):.3f}")
        print(f"   anomaly frames by Tier: {dict(tier_fr)}")
        print(f"   number of frames containing each OOD class: {dict(sorted(cls_fr.items(),key=lambda x:-x[1]))}")
if __name__=="__main__": main()
