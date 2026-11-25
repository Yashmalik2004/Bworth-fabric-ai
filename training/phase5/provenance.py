"""Provenance groups: the unit of independence for every Phase 5 split.

Fabrics are linked (transitively) when they share a non-empty provenance_group, source_product_id or evidence
product_identifier (product ids are normalised: case, whitespace). All photographs of all fabrics in a group stay
in one split.
"""

from __future__ import annotations

import pandas as pd


def norm(x) -> str:
    return " ".join(str(x).strip().casefold().split())


def build_groups(fabrics: pd.DataFrame, verification: pd.DataFrame, fabric_ids) -> pd.DataFrame:
    ids = sorted(set(fabric_ids))
    parent = {f: f for f in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    keys: dict[str, list[str]] = {}
    fab = fabrics[fabrics.fabric_id.isin(ids)]
    for f in fab.itertuples():
        if norm(getattr(f, "provenance_group", "")):
            keys.setdefault("pg:" + norm(f.provenance_group), []).append(f.fabric_id)
        if norm(getattr(f, "source_product_id", "")):
            keys.setdefault("pid:" + norm(f.source_product_id), []).append(f.fabric_id)
    ver = verification[verification.fabric_id.isin(ids)]
    for r in ver.itertuples():
        if norm(getattr(r, "product_identifier", "")):
            keys.setdefault("pid:" + norm(r.product_identifier), []).append(r.fabric_id)
    for members in keys.values():
        for m in members[1:]:
            ra, rb = find(members[0]), find(m)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    links: dict[str, set] = {f: set() for f in ids}
    for k, members in keys.items():
        if len(set(members)) > 1:
            for m in members:
                links[m].add(k)
    out = pd.DataFrame({"fabric_id": ids})
    out["provenance_id"] = ["PROV-" + find(f) for f in ids]
    out["linked_by"] = [";".join(sorted(links[f])) for f in ids]
    out["group_size"] = out.groupby("provenance_id").fabric_id.transform("size")
    return out
