#!/usr/bin/env python
"""
OffRoad-OOD benchmark — GOOSE feasibility + leakage check (SPLIT_PROTOCOL §7.8)
Based on the real GOOSE 2D structure:
  labels/<split>/<scenario>/<scenario>__<frame>_<ts>_labelids.png  (mode L, pixel=label_key 0-63)
  goose_label_mapping.csv: class_name,label_key,has_instance,hex
Usage: module load scicomp-python-env/2025.2; python audit_goose.py [val|train|both]
Output: terminal report + data/goose/audit_goose.json
No assumptions — all statistics come from real labels.
"""
import os, sys, glob, json, csv
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import yaml

from paths import ROOT
GOOSE = f"{ROOT}/data/goose/extracted"
ONTO = f"{ROOT}/configs/meta_ontology.yaml"
CSV  = f"{GOOSE}/goose_label_mapping.csv"
MIN_OOD_PX = 10          # minimum pixels for a frame to contain OOD (removes single-pixel noise)
MIN_SUPPORT_FRAMES = 50  # per-class single-column threshold (protocol v3-6)
NUMCLASS = 64

def load_mapping():
    """label_key -> class_name, and validate consistency with meta_ontology. Returns key->(name, role, tier, is_border, group)."""
    key2name = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            key2name[int(r["label_key"])] = r["class_name"]
    onto = yaml.safe_load(open(ONTO))["GOOSE"]["classes"]
    # consistency check
    csv_names = set(key2name.values()); onto_names = set(onto.keys())
    missing = csv_names - onto_names      # in csv but not mapped in yaml -> fatal
    extra   = onto_names - csv_names      # in yaml but not in csv -> spelling error
    role_of = {}
    TIER = {"ood_tier1":"T1","ood_tier2":"T2","ood_tier3":"T3"}
    for k, name in key2name.items():
        spec = onto.get(name, {})
        role = spec.get("role", "UNMAPPED")
        is_border = role == "borderline"
        eff = spec.get("default_role", "id_terrain") if is_border else role
        tier = TIER.get(eff, "")
        role_of[k] = dict(name=name, role=eff, raw_role=role, tier=tier,
                          is_border=is_border, group=spec.get("group",""),
                          catch_all=spec.get("catch_all", False))
    return key2name, role_of, sorted(missing), sorted(extra)

def is_ood(r):  return r["role"].startswith("ood_")
def is_id(r):   return r["role"].startswith("id_")
def is_ign(r):  return r["role"] == "ignore"

def scan_split(split, role_of):
    files = sorted(glob.glob(f"{GOOSE}/labels/{split}/*/*_labelids.png"))
    if not files:
        return None
    cls_px = np.zeros(NUMCLASS, dtype=np.int64)     # total pixels per class
    cls_fr = np.zeros(NUMCLASS, dtype=np.int64)     # number of frames each class appears in
    per_scen = defaultdict(lambda: dict(frames=0, ood_frames=0, ood_classes=Counter()))
    ood_class_frames = Counter()                     # number of frames containing each OOD class (>=MIN_OOD_PX)
    tier_frames = Counter()
    n_anom_frames = n_pure_id = 0
    ign_fracs, ood_fracs = [], []
    bad_vals = Counter()
    for fp in files:
        scen = os.path.basename(os.path.dirname(fp))
        a = np.asarray(Image.open(fp))
        bc = np.bincount(a.ravel(), minlength=256)
        if bc[NUMCLASS:].sum() > 0:                  # anomaly: pixel value > 63 appears
            for v in np.nonzero(bc[NUMCLASS:])[0]: bad_vals[v+NUMCLASS]+=int(bc[v+NUMCLASS])
        bc = bc[:NUMCLASS]; tot = int(a.size)
        present = np.nonzero(bc)[0]
        cls_px += bc; cls_fr[present] += 1
        ign_px = sum(int(bc[k]) for k in present if is_ign(role_of[k]))
        ood_px = sum(int(bc[k]) for k in present if is_ood(role_of[k]))
        ign_fracs.append(ign_px/tot); ood_fracs.append(ood_px/tot)
        per_scen[scen]["frames"] += 1
        frame_ood_tiers=set(); has_ood=False
        for k in present:
            r=role_of[k]
            if is_ood(r) and bc[k] >= MIN_OOD_PX:
                has_ood=True; ood_class_frames[r["name"]]+=1
                frame_ood_tiers.add(r["tier"]); per_scen[scen]["ood_classes"][r["name"]]+=1
        for t in frame_ood_tiers: tier_frames[t]+=1
        if has_ood: n_anom_frames+=1; per_scen[scen]["ood_frames"]+=1
        elif ood_px==0: n_pure_id+=1
    return dict(split=split, n_frames=len(files), cls_px=cls_px, cls_fr=cls_fr,
                per_scen=per_scen, ood_class_frames=ood_class_frames, tier_frames=tier_frames,
                n_anom_frames=n_anom_frames, n_pure_id=n_pure_id,
                ign_fracs=ign_fracs, ood_fracs=ood_fracs, bad_vals=dict(bad_vals))

def pct(x):
    x=np.array(x);
    return {q: round(float(np.percentile(x,q)),4) for q in (50,75,90,95,99)}

def report(splits, key2name, role_of, missing, extra):
    print("="*70)
    print("META-ONTOLOGY ↔ goose_label_mapping.csv consistency check")
    print("="*70)
    print(f"  CSV classes={len(key2name)}  meta mapped classes={len(set(r['name'] for r in role_of.values()))}")
    print(f"  🔴 in csv but not mapped in yaml (must fix): {missing if missing else 'none ✓'}")
    print(f"  🟡 in yaml but not in csv (spelling error?): {extra if extra else 'none ✓'}")
    unmapped=[role_of[k]['name'] for k in role_of if role_of[k]['raw_role']=='UNMAPPED']
    print(f"  UNMAPPED: {unmapped if unmapped else 'none ✓'}")

    out={"validation":{"missing":missing,"extra":extra}, "splits":{}}
    agg=None
    for S in splits:
        if S is None: continue
        print("\n"+"="*70); print(f"SPLIT = {S['split']}  ({S['n_frames']} frames)"); print("="*70)
        if S["bad_vals"]: print(f"  ⚠️ illegal pixel value > 63 appears: {S['bad_vals']}")
        print(f"  anomaly frames(≥{MIN_OOD_PX}px OOD): {S['n_anom_frames']}  | pure ID frames: {S['n_pure_id']}  "
              f"| {'✅≥100 eligible for main leaderboard' if S['n_anom_frames']>=100 else '🔴<100 must downgrade'}")
        print(f"  anomaly frames by Tier: T1={S['tier_frames'].get('T1',0)} T2={S['tier_frames'].get('T2',0)} T3={S['tier_frames'].get('T3',0)}")
        print(f"  ignore pixel fraction P50/75/90/95/99: {pct(S['ign_fracs'])}  | frames>40%: {sum(f>0.4 for f in S['ign_fracs'])}")
        print(f"  OOD pixel fraction   P50/75/90/95/99: {pct(S['ood_fracs'])}")
        print(f"  --- number of frames containing each OOD class (long-tail, 🔴<{MIN_SUPPORT_FRAMES} single-column unstable) ---")
        for name,c in sorted(S['ood_class_frames'].items(), key=lambda x:-x[1]):
            print(f"      {c:5d}  {name}{'   🔴' if c<MIN_SUPPORT_FRAMES else ''}")
        print(f"  --- per scenario: frame count / anomaly frames / OOD class count ---")
        for scen,d in sorted(S['per_scen'].items()):
            print(f"      {d['frames']:4d}f {d['ood_frames']:4d}ood {len(d['ood_classes']):2d}cls  {scen}")
        out["splits"][S['split']]=dict(n_frames=S['n_frames'], n_anom_frames=S['n_anom_frames'],
            n_pure_id=S['n_pure_id'], tier_frames=dict(S['tier_frames']),
            ood_class_frames=dict(S['ood_class_frames']),
            ign_pct=pct(S['ign_fracs']), ood_pct=pct(S['ood_fracs']),
            scenarios={s:{"frames":d["frames"],"ood_frames":d["ood_frames"],
                          "ood_classes":dict(d["ood_classes"])} for s,d in S['per_scen'].items()},
            bad_vals=S["bad_vals"])

    # cross-split scenario/location leakage check
    real=[s for s in splits if s]
    if len(real)>=2:
        print("\n"+"="*70); print("leakage check: cross-split scenario/location overlap"); print("="*70)
        scen_sets={s['split']:set(s['per_scen'].keys()) for s in real}
        loc_sets ={s['split']:set('_'.join(k.split('_')[3:]) for k in s['per_scen'].keys()) for s in real}
        names=list(scen_sets)
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                a,b=names[i],names[j]
                so=scen_sets[a]&scen_sets[b]; lo=loc_sets[a]&loc_sets[b]
                print(f"  {a} ∩ {b}: scenario overlap={sorted(so) if so else 'none✓'}")
                print(f"  {a} ∩ {b}: location overlap={sorted(lo) if lo else 'none✓'}  {'🔴 official split location crosses boundary -> must re-split for isolation' if lo else ''}")
                out.setdefault("leakage",{})[f"{a}_vs_{b}"]={"scenario_overlap":sorted(so),"location_overlap":sorted(lo)}

    json.dump(out, open(f"{ROOT}/data/goose/audit_goose.json","w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {ROOT}/data/goose/audit_goose.json")

if __name__=="__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "both"
    splits = ["val","train"] if which=="both" else [which]
    key2name, role_of, missing, extra = load_mapping()
    scanned=[scan_split(s, role_of) for s in splits]
    found=[s for s in scanned if s]
    if not found:
        print("No labelids files found, check whether extraction is complete:", splits); sys.exit(1)
    report(scanned, key2name, role_of, missing, extra)
