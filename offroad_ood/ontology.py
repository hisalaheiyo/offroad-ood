#!/usr/bin/env python
"""
OffRoad-OOD — ontology parsing (single source of truth for dataloader + evaluation)
Built from configs/meta_ontology.yaml + goose_label_mapping.csv:
  - label_key(0-63) -> role / tier / group
  - contiguous train_id renumbering of ID classes (deterministic: ascending by label_key)
  - training label_key -> train_id (OOD/ignore/borderline-as-ignore -> IGNORE=255)
  - test label_key -> {1:OOD, 0:ID, 255:ignore}
borderline handled by default_role by default (rock/asphalt -> id_terrain), leaving a confusion-matrix adjudication hook.
"""
import os, csv, yaml
from paths import ROOT
IGNORE = 255
TIERMAP = {"ood_tier1":"T1","ood_tier2":"T2","ood_tier3":"T3"}

def _eff_role(spec):
    """borderline -> its default_role; otherwise the original role value."""
    role = spec.get("role","UNMAPPED")
    if role == "borderline":
        return spec.get("default_role","id_terrain"), True
    return role, False

def load_goose_ontology(onto_path=None, csv_path=None):
    onto_path = onto_path or f"{ROOT}/configs/meta_ontology.yaml"
    csv_path  = csv_path  or f"{ROOT}/data/goose/extracted/goose_label_mapping.csv"
    onto = yaml.safe_load(open(onto_path))["GOOSE"]["classes"]
    key2name = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            key2name[int(r["label_key"])] = r["class_name"]

    info = {}        # key -> dict(name, eff_role, tier, group, is_border, catch_all)
    missing = []
    for k, name in key2name.items():
        spec = onto.get(name)
        if spec is None:
            missing.append(name);
            info[k] = dict(name=name, eff_role="UNMAPPED", tier="", group="", is_border=False, catch_all=False)
            continue
        eff, isb = _eff_role(spec)
        info[k] = dict(name=name, eff_role=eff, tier=TIERMAP.get(eff,""),
                       group=spec.get("group",""), is_border=isb,
                       catch_all=spec.get("catch_all", False))
    if missing:
        raise ValueError(f"meta_ontology missing mapping (fatal): {sorted(missing)}")

    # ID classes -> contiguous train_id (deterministic: ascending label_key)
    id_keys = sorted(k for k,v in info.items() if v["eff_role"].startswith("id_"))
    key2trainid = {k:i for i,k in enumerate(id_keys)}
    num_train = len(id_keys)

    # training mapping table: label_key -> train_id or IGNORE
    train_lut = {k:(key2trainid[k] if k in key2trainid else IGNORE) for k in key2name}
    # test mapping table: label_key -> 1(OOD)/0(ID)/255(ignore)
    def test_role(v):
        if v["eff_role"].startswith("ood_"): return 1
        if v["eff_role"].startswith("id_"):  return 0
        return IGNORE
    test_lut = {k:test_role(v) for k,v in info.items()}

    return dict(info=info, key2name=key2name, key2trainid=key2trainid, num_train=num_train,
                train_lut=train_lut, test_lut=test_lut, id_keys=id_keys,
                ood_keys=sorted(k for k,v in info.items() if v["eff_role"].startswith("ood_")),
                ignore_keys=sorted(k for k,v in info.items() if v["eff_role"]=="ignore"))

def _build_from_keymap(key2name, name2spec, num_keys):
    """Generic: given key->name and name->spec, build info/lut/train_id etc. (same logic as GOOSE)."""
    info = {}; missing = []
    for k, name in key2name.items():
        spec = name2spec.get(name)
        if spec is None:
            missing.append(name); info[k] = dict(name=name, eff_role="UNMAPPED", tier="", group="", is_border=False, catch_all=False); continue
        eff, isb = _eff_role(spec)
        info[k] = dict(name=name, eff_role=eff, tier=TIERMAP.get(eff,""), group=spec.get("group",""),
                       is_border=isb, catch_all=spec.get("catch_all", False))
    if missing: raise ValueError(f"meta_ontology missing mapping: {sorted(missing)}")
    id_keys = sorted(k for k,v in info.items() if v["eff_role"].startswith("id_"))
    key2trainid = {k:i for i,k in enumerate(id_keys)}
    train_lut = {k:(key2trainid[k] if k in key2trainid else IGNORE) for k in key2name}
    def trole(v): return 1 if v["eff_role"].startswith("ood_") else (0 if v["eff_role"].startswith("id_") else IGNORE)
    test_lut = {k:trole(v) for k,v in info.items()}
    return dict(info=info, key2name=key2name, key2trainid=key2trainid, num_train=len(id_keys),
                train_lut=train_lut, test_lut=test_lut, id_keys=id_keys,
                ood_keys=sorted(k for k,v in info.items() if v["eff_role"].startswith("ood_")),
                ignore_keys=sorted(k for k,v in info.items() if v["eff_role"]=="ignore"))

def load_wildscenes_ontology(onto_path=None):
    onto_path = onto_path or f"{ROOT}/configs/meta_ontology.yaml"
    cbi = yaml.safe_load(open(onto_path))["WildScenes"]["classes_by_index"]
    key2name = {int(k):v["name"] for k,v in cbi.items()}
    name2spec = {v["name"]:{kk:vv for kk,vv in v.items() if kk!="name"} for v in cbi.values()}
    return _build_from_keymap(key2name, name2spec, max(key2name)+1)

def load_rellis_ontology(onto_path=None):
    onto_path = onto_path or f"{ROOT}/configs/meta_ontology.yaml"
    cbi = yaml.safe_load(open(onto_path))["RELLIS-3D"]["classes_by_index"]
    key2name = {int(k):v["name"] for k,v in cbi.items()}
    name2spec = {v["name"]:{kk:vv for kk,vv in v.items() if kk!="name"} for v in cbi.values()}
    return _build_from_keymap(key2name, name2spec, max(key2name)+1)

def load_rugd_ontology(onto_path=None):
    onto_path = onto_path or f"{ROOT}/configs/meta_ontology.yaml"
    cbi = yaml.safe_load(open(onto_path))["RUGD"]["classes_by_index"]
    key2name = {int(k):v["name"] for k,v in cbi.items()}
    name2spec = {v["name"]:{kk:vv for kk,vv in v.items() if kk!="name"} for v in cbi.values()}
    return _build_from_keymap(key2name, name2spec, max(key2name)+1)

def load_competitor_ontology(base, onto_path=None):
    """Competitor (Anomalies-by-Synthesis) comparison track: OOD only={vehicle,building,person}, all other non-ignore→ID
    (including man-made non-anomalous classes such as pole/fence/barrier→id_terrain), void/catch_all→ignore. base='rellis'/'rugd'.
    Reproduces their OOD definition for a fair comparison against their published AP."""
    onto_path = onto_path or f"{ROOT}/configs/meta_ontology.yaml"
    key = "RELLIS-3D" if base=="rellis" else "RUGD"
    d = yaml.safe_load(open(onto_path))[key]
    cbi = d["classes_by_index"]; comp_ood = set(d["competitor_split"]["ood"])
    key2name = {int(k):v["name"] for k,v in cbi.items()}
    name2spec = {}
    for v in cbi.values():
        name=v["name"]
        if name in comp_ood: spec={"role":"ood_tier1"}                       # competitor OOD
        elif v.get("role")=="ignore" or v.get("catch_all"): spec={"role":"ignore"}
        else:
            r=v.get("role","id_terrain")
            if r=="borderline": r=v.get("default_role","id_terrain")
            if r.startswith("ood"): r="id_terrain"                            # man-made non-anomalous→ID(normal)
            spec={"role":r}
        name2spec[name]=spec
    return _build_from_keymap(key2name, name2spec, max(key2name)+1)

if __name__ == "__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="wildscenes":
        o=load_wildscenes_ontology()
        print(f"[WildScenes] train ID classes={o['num_train']} OOD={len(o['ood_keys'])} ignore={len(o['ignore_keys'])} coverage{o['num_train']+len(o['ood_keys'])+len(o['ignore_keys'])}/{len(o['key2name'])}")
        print("ID:", [o['info'][k]['name'] for k in o['id_keys']])
        print("OOD:", [(o['info'][k]['name'],o['info'][k]['tier']) for k in o['ood_keys']])
        print("ignore:", [o['info'][k]['name'] for k in o['ignore_keys']])
        sys.exit(0)
    o = load_goose_ontology()
    print(f"train ID class count={o['num_train']}  OOD class count={len(o['ood_keys'])}  ignore class count={len(o['ignore_keys'])}")
    print("ID train classes (name@key->train_id):")
    for k in o["id_keys"]:
        print(f"  {o['key2trainid'][k]:2d} <- key{k:2d} {o['info'][k]['name']}")
    print("OOD classes (by tier):")
    from collections import defaultdict
    bt=defaultdict(list)
    for k in o["ood_keys"]: bt[o['info'][k]['tier']].append(o['info'][k]['name'])
    for t in ("T1","T2","T3"): print(f"  {t}: {sorted(bt[t])}")
    # sanity: coverage
    tot=len(o['key2name']); cov=o['num_train']+len(o['ood_keys'])+len(o['ignore_keys'])
    print(f"coverage check: {cov}/{tot} {'✅' if cov==tot else '🔴 missing'}")
