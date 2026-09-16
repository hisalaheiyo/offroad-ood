#!/usr/bin/env python
"""
GOOSE site-level leakage-free re-split design (SPLIT_PROTOCOL §7.7-A)
Reads data/goose/audit_goose.json, merges the train+val parts of each scenario, groups by location (site),
proposes a whole-site train/val/test ~70/10/20 (by frame) partition, validates test anomaly frames≥100 + Tier coverage.
Sites are matched by known location keywords; uncertain ones are marked AMBIGUOUS for manual review.
Usage: module load scicomp-python-env/2025.2; python plan_split.py
"""
import json, sys
from collections import defaultdict
ROOT="."
J=json.load(open(f"{ROOT}/data/goose/audit_goose.json"))

# known physical location keywords (substring match); note touareg/tiguan are vehicle names, not locations
SITES=["garching","aying","neubiberg","campus","siegertsbrunn","hoehenkirchner",
       "putzbrunn","solalinden","hohenbrunn","neuperlach","flight"]
TIER_GROUPS={"T1":{"car","truck","bus","motorcycle","bicycle","caravan","trailer","heavy_machinery",
                   "on_rails","kick_scooter","military_vehicle","building","wall","fence","bridge",
                   "container","tunnel","guard_rail","wire","person","rider"},
             "T3":{"animal"}}  # remaining OOD classes go to T2
def tier_of(cls):
    if cls in TIER_GROUPS["T3"]: return "T3"
    if cls in TIER_GROUPS["T1"]: return "T1"
    return "T2"

def site_of(scen):
    rest=scen.split("_",1)[1] if "_" in scen else scen   # strip date
    hits=[s for s in SITES if s in scen]
    return hits[0] if len(hits)==1 else (hits[0]+"?MULTI" if hits else "AMBIGUOUS:"+rest)

# merge train+val for each scenario
merged={}  # scen -> {frames, ood_frames, classes:Counter}
for split in ("train","val"):
    for scen,d in J["splits"].get(split,{}).get("scenarios",{}).items():
        m=merged.setdefault(scen, {"frames":0,"ood_frames":0,"classes":defaultdict(int)})
        m["frames"]+=d["frames"]; m["ood_frames"]+=d["ood_frames"]
        for c,n in d["ood_classes"].items(): m["classes"][c]+=n

# group into sites
sites=defaultdict(lambda:{"scens":[], "frames":0,"ood_frames":0,"classes":set()})
for scen,m in merged.items():
    s=site_of(scen); g=sites[s]
    g["scens"].append(scen); g["frames"]+=m["frames"]; g["ood_frames"]+=m["ood_frames"]
    g["classes"]|=set(m["classes"].keys())

print(f"=== {len(merged)} unique scenarios -> {len(sites)} sites ===")
for s,g in sorted(sites.items(), key=lambda x:-x[1]["frames"]):
    flag=" ⚠️" if ("AMBIGUOUS" in s or "MULTI" in s) else ""
    print(f"  {g['frames']:5d}f {g['ood_frames']:5d}ood {len(g['classes']):2d}cls {len(g['scens'])}scen  {s}{flag}")
    for sc in sorted(g["scens"]): print(f"          - {sc}")

# ---- explicit terrain-aware design (GOOSE is from Bundeswehr Univ Munich, Neubiberg campus) ----
TERRAIN={  # semi-urban vs pure off-road
 "campus":"semi-urban","neubiberg":"semi-urban","neuperlach":"semi-urban",
 "garching":"mil-training","aying":"rural","flight":"uncertain",
 "solalinden":"off-road","siegertsbrunn":"off-road","hoehenkirchner":"off-road",
 "putzbrunn":"off-road","hohenbrunn":"off-road"}
DESIGN={  # whole site goes into the same split, eliminating same-drive leakage; semi-urban→train preserves test off-road purity
 "campus":"train","neubiberg":"train","neuperlach":"train",  # semi-urban→train
 "aying":"train","garching":"train",                          # large rural/training grounds→train
 "flight":"val","putzbrunn":"val",                            # →val (flight terrain unclear, put in val)
 "solalinden":"test","siegertsbrunn":"test","hoehenkirchner":"test","hohenbrunn":"test"}  # pure off-road field/forest tracks→test
TOT=sum(g["frames"] for g in sites.values())
assign={"train":[],"val":[],"test":[]}
for s in sites:
    sp=DESIGN.get(s.split("?")[0].split(":")[0])
    if sp: assign[sp].append(s)
    else: print(f"  ⚠️ site not assigned in DESIGN: {s}")

print("\n=== finalized partition (terrain-aware, whole-site, zero leakage) ===")
def pid(s): return sites[s]["frames"]-sites[s]["ood_frames"]   # pure ID frames (FPR negative class)
for k in ("train","val","test"):
    fr=sum(sites[s]["frames"] for s in assign[k]); ood=sum(sites[s]["ood_frames"] for s in assign[k])
    purid=sum(pid(s) for s in assign[k])
    cls=set().union(*[sites[s]["classes"] for s in assign[k]]) if assign[k] else set()
    tiers=sorted({tier_of(c) for c in cls})
    ok_ood="✅" if (k!="test" or ood>=100) else "🔴<100"
    ok_tier="✅" if tiers==["T1","T2","T3"] else f"⚠️ missing {set(['T1','T2','T3'])-set(tiers)}"
    ok_pid="✅" if purid>=30 else f"🔴 pure ID only {purid} (insufficient FPR negative class)"
    print(f"  [{k:5s}] {fr}f({fr/TOT*100:.0f}%) | anomaly {ood}f {ok_ood} | pure ID {purid}f {ok_pid} | {len(cls)} classes Tier{tiers}{ok_tier}")
    for s in sorted(assign[k]): print(f"          {sites[s]['frames']:5d}f {sites[s]['ood_frames']:4d}ood [{TERRAIN.get(s,'?')}] {s}")

# validation: no site crosses splits + source of test T3
allsites=[s for k in assign for s in assign[k]]
print(f"\n  zero-leakage check: {len(allsites)} sites assigned, unique={len(set(allsites))} {'✅ no overlap' if len(allsites)==len(set(allsites)) else '🔴 overlap'}")
import json as _j
_j.dump({k:assign[k] for k in assign}, open(f"{ROOT}/configs/goose_site_split.json","w"), indent=2)
print(f"  [saved] configs/goose_site_split.json")
