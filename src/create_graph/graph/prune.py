def prune_edges(edges, max_out_degree=3, max_dist_px=None, mutual_confirmation=True):
    if not edges:
        return []

    # 0) 去掉同向重复：保留 dist 最短的那条
    best = {}
    for e in edges:
        key = (e["src"], e["dst"])
        if key not in best or e["dist"] < best[key]["dist"]:
            best[key] = e
    edges = list(best.values())

    # 1) 距离阈值（欧氏距离）
    if max_dist_px is not None:
        edges = [e for e in edges if e["dist"] <= max_dist_px]
    if not edges:
        return []

    # 2) 互认：显式要求 (i,j) 和 (j,i) 都存在
    if mutual_confirmation:
        pairs = {(e["src"], e["dst"]) for e in edges}
        mutual = { (i,j) for (i,j) in pairs if (j,i) in pairs }
        edges = [e for e in edges if (e["src"], e["dst"]) in mutual]
        if not edges:
            return []

    # 3) 限制每个 src 的出度（按 dist 由小到大取前 k）
    out_map = {}
    for e in edges:
        out_map.setdefault(e["src"], []).append(e)

    pruned = []
    for src, lst in out_map.items():
        lst.sort(key=lambda x: x["dist"])
        pruned.extend(lst[:max_out_degree])

    return pruned
