#!/usr/bin/env python
"""Dataset selection helper: name -> (DataloaderClass, ontology_loader). All datasets share the same interface, and baseline scripts are parameterized accordingly."""
def get_dataset(name="goose"):
    if name=="goose":
        from goose_dataset import GooseOOD
        from ontology import load_goose_ontology
        return GooseOOD, load_goose_ontology
    if name=="wildscenes":
        from wildscenes_dataset import WildScenesOOD
        from ontology import load_wildscenes_ontology
        return WildScenesOOD, load_wildscenes_ontology
    if name=="rellis":
        from rellis_dataset import RellisOOD
        from ontology import load_rellis_ontology
        return RellisOOD, load_rellis_ontology
    if name=="rugd":
        from rugd_dataset import RugdOOD
        from ontology import load_rugd_ontology
        return RugdOOD, load_rugd_ontology
    if name=="rellis_comp":
        from competitor_dataset import RellisCompOOD
        from ontology import load_competitor_ontology
        return RellisCompOOD, (lambda: load_competitor_ontology("rellis"))
    if name=="rugd_comp":
        from competitor_dataset import RugdCompOOD
        from ontology import load_competitor_ontology
        return RugdCompOOD, (lambda: load_competitor_ontology("rugd"))
    if name=="goose_v2":
        from goose_dataset import GooseV2OOD
        from ontology import load_goose_ontology
        return GooseV2OOD, load_goose_ontology
    if name=="goose_v3":
        from goose_dataset import GooseV3OOD
        from ontology import load_goose_ontology
        return GooseV3OOD, load_goose_ontology
    raise ValueError(f"unknown dataset: {name}")
