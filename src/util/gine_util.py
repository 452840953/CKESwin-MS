"""Direct helpers extracted from the original shared utilities."""
import re
import numpy as np
import torch


def load_specimens_from_txt(txt_path):
    """读取 test.txt，返回标本号集合和映射关系"""
    specimens = set()
    mapping = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in re.split(r"[,\t]", line) if x.strip()]
            spn = parts[0].lower()
            clsname = parts[1] if len(parts) >= 2 else ""
            specimens.add(spn)
            if clsname:
                mapping[spn] = clsname
    return specimens, mapping

def path_contains_any_specimen(img_path: str, specimen_set: set[str]) -> tuple[bool, str]:
    """判断路径中是否包含任意标本号"""
    p = (img_path or "").lower()
    for spn in specimen_set:
        if spn in p:
            return True, spn
    return False, ""

def count_by_class(indices, dataset, idx_to_class=None):
    ys = [int(dataset[i].y.item()) for i in indices]
    counts = np.bincount(ys, minlength=len(idx_to_class) if idx_to_class else (max(ys) + 1))
    rows = []
    total = counts.sum()
    for cid, c in enumerate(counts):
        cname = idx_to_class.get(cid, str(cid)) if idx_to_class else str(cid)
        rows.append((cname, int(c), (c / total if total > 0 else 0.0)))
    return rows, int(total)

def pretty_table(rows, title):
    lines = [title, "  类别\t数量\t占比"]
    for cname, c, r in rows:
        lines.append(f"  {cname}\t{c}\t{r:.2%}")
    return "\n".join(lines)
